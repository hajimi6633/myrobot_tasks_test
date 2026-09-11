"""充电枪 6 阶段任务：显式 Phase 状态机（Action Server，主循环）。

迁移映射：src/env/tasks/charging_phases.py（闭包 st 状态）→ 本文件。
重构收益（对话第 1、6 轮）：
  - 闭包 st dict → PhaseData 显式字段：可挂起 / 恢复 / 单元测试
  - 散落常量 → declare_parameter（launch YAML 可覆盖）
  - 诊断打印 → self.log（后续可拆独立 DiagnosticsNode 订阅反馈）
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from rclike import Node, ActionServer, GoalState


@dataclass
class PhaseData:
    """单阶段运行时状态（原 charging_phases 闭包 st 的显式化）。"""
    name: str
    desc: str = ""
    grip: float = 1.0                     # 阶段夹爪开合
    trajectory: object | None = None     # Trajectory；None=自由控制段
    nominal: np.ndarray | None = None     # 导纳名义目标
    prev_q_des: np.ndarray | None = None  # 链式 IK 初值（连续两步解连续）
    prev_actual: np.ndarray | None = None  # 目标变化限幅基准
    blocked_n: int = 0                   # 停滞计数（窗口式累计推进）
    phase_done: bool = False
    phase_msg: str = ""


class ChargingAction(Node):
    """6 阶段：抓取 → 移动 → 插枪(导纳) → 拔枪 → 归位 → 复位。"""

    # weld / site 名称集中管理（迁移自 ChargingGunTask 类常量）
    EQ_SOCKET, EQ_GUN_EE = "eq_socgun_1", "eq_gun_ee"
    GUN_BODY, GRIPPER_BASE = "charging_gun_1", "carry_shell"
    SOCKET_BODY = "charing_socket"

    def __init__(self, bus, clock, ik=None, grasp=None, pose=None):
        super().__init__("charging_action", bus, clock)
        self.ik, self.grasp, self.pose = ik, grasp, pose   # launch 回填
        # ---- 参数化：原 charging_phases 顶部常量全部迁到这里 ----
        self.declare_parameter("f_block", 40.0, "插枪推进力预算 (N)")
        self.declare_parameter("insert_step", 0.002, "名义推进步长 (m)")
        self.declare_parameter("align_tol", 0.0015, "对心阈值 (m)")
        self.declare_parameter("align_step", 0.001, "对心闭环步长 (m)")
        self.declare_parameter("done_dist", 0.006, "到底距离阈值 (m)")
        self.declare_parameter("max_travel", 0.005, "闭环段 IK 行程硬限 (rad)")
        self.phases: list[PhaseData] = []
        self._pi = 0                     # 当前阶段索引
        self.action = ActionServer(self, "charging", self._step)
        # 安全慢通道：zone 变化 → 挂起 / 恢复（快通道由 ArmController 冻结）
        self.create_subscription("/safety_state", self._on_safety)

    # ---- 阶段构建（TODO 迁移：build_phase1~6 的规划与导纳逻辑）----
    def _build_phases(self):
        names = ["1_grasp", "2_move", "3_insert_admittance",
                 "4_pull", "5_home_admittance", "6_reset"]
        self.phases = [PhaseData(name=n) for n in names]
        self._pi = 0

    # ---- 每拍推进（Executor → on_tick → action.spin → _step）----
    def _step(self, handle):
        """单步推进当前阶段；返回 None 继续，返回终止态结束 goal。
        SUSPENDED 期间 ActionServer.spin 不会调用本函数（冻结在当前阶段）。"""
        ph = self.phases[self._pi]
        # TODO 迁移（示例骨架，逻辑见 charging_phases.py 各 build_phaseN）：
        #   1. 轨迹段：q_des = ph.trajectory.at(i)
        #   2. 导纳段：读 /wrench → 对心修正 → IK 服务（retry=False,
        #      max_travel=self.get_parameter("max_travel")）
        #   3. done_condition：提前结束 / 停滞保护
        #   4. 阶段切换：grasp.attach/detach 服务 + pose.handoff
        self.action.publish_feedback({"phase": ph.name, "index": self._pi})
        return None

    def on_tick(self):
        self.action.spin()

    # ---- 安全慢通道：挂起 / 恢复 ----
    def _on_safety(self, msg, stamp):
        if msg["zone"] == "stop":
            self.action.suspend()
        elif self.action.handle is not None and \
                self.action.handle.state.name == "SUSPENDED":
            # TODO: 恢复前人工确认策略（或直接自动恢复）
            self.action.resume()

    def send_goal(self, goal=None):
        self._build_phases()
        return self.action.send_goal(goal or {})
