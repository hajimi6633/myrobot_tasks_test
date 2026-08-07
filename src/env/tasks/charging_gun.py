"""充电枪抓取-插拔长序列任务（6 阶段，解耦设计）。

阶段流程：
  1. 抓取充电枪（gun_site_2 → gun_site_1 → 闭合 → 解除插座 weld → 耦合到末端）
  2. 移动（gun_site → charing_site_2 → car_site_2）
  3. 插枪（导纳控制，遇阻调整，到底检测）
  4. 拔枪（gun_site → car_site_2 → charing_site_2）
  5. 枪体归位（导纳控制，gun_site → charing_site_1 → 下移 → 激活插座 weld）
  6. 机械臂复位（松开夹爪 → 回 home）

每个阶段封装为 Phase dataclass，独立 setup/step/exit，便于后续插入新阶段。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import mujoco

from src.env.tasks.base import BaseTask


# ---- 阶段数据结构 ----
@dataclass
class PhaseContext:
    """阶段间共享的上下文（传感器、控制器、偏移量等）。"""
    env: object           # ArmEnv
    ik: object            # IKSolver
    task: "ChargingGunTask"
    force_sensor: object  # ForceTorqueSensor
    constraints: object   # ConstraintManager
    coupler: object       # GraspCoupler
    admittance: object    # AdmittanceController
    dt_ctrl: float = 0.05
    # gun_site 的 site id，抓取后用 site_id 直接对 gun_site 求 IK，
    # 利用枪体延伸臂展，避免 gun↔ee 偏移转换导致 ee 目标超出工作空间
    gun_site_id: int = -1
    # 抓取后 gun_site 相对 ee_link 的偏移，存于 ee_link 局部坐标系（恒定）。
    # 因枪体经 GraspCoupler 刚性耦合到 gripper_base，ee_link 与 gripper_base
    # 刚性连接，故该局部偏移不随机械臂姿态变化。转换时用当前 ee_link 旋转
    # 矩阵旋转回世界系：ee_target = gun_target - R_ee @ offset_local。
    ee_to_gun_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # 抓取时 gun_site 相对 ee_link 的旋转 R_gun_ee = R_gun @ R_ee.T，
    # 抓取后恒定，用于将 gun 目标姿态转换为 ee_link 目标姿态（确保 z 轴对齐）
    grasp_rot: np.ndarray = field(default_factory=lambda: np.eye(3))
    # 阶段运行时数据
    forces: list = field(default_factory=list)
    # 阶段提前结束标志（如插枪到底），由 on_step/done_condition 设置
    phase_done: bool = False
    phase_msg: str = ""


@dataclass
class Phase:
    """一个独立阶段。执行器按 name/enter/step/exit 顺序驱动。"""
    name: str
    desc: str = ""
    trajectory: object | None = None   # Trajectory；None=自由控制段（用 n_steps）
    n_steps: int = 0                   # 自由控制段步数
    grip_ratio: float = 1.0
    on_enter: Callable[[PhaseContext], None] | None = None
    # on_step(ctx, idx) -> np.ndarray | None；返回 q_des 覆盖轨迹，None=用轨迹
    on_step: Callable[[PhaseContext, int], object] | None = None
    on_exit: Callable[[PhaseContext], None] | None = None
    # 阶段完成判定，返回 True 提前结束当前阶段
    done_condition: Callable[[PhaseContext], bool] | None = None
    monitor_force: bool = False        # 记录每步力到 ctx.forces
    use_admittance: bool = False       # 该段用导纳修正目标位置

    @property
    def length(self) -> int:
        if self.trajectory is not None:
            return len(self.trajectory)
        return self.n_steps


class ChargingGunTask(BaseTask):
    """充电枪任务：管理 site/body 名称与标记点。阶段逻辑见 charging_phases。"""

    # body / site 名称集中管理，便于替换模型时统一修改
    GUN_BODY = "charging_gun_1"
    SOCKET_BODY = "charing_socket"
    CAR_SOCKET_BODY = "car_socket"
    GRIPPER_BASE_BODY = "gripper_base"
    GUN_SITE = "gun_site"
    GUN_SITE_1 = "gun_site_1"
    GUN_SITE_2 = "gun_site_2"
    CHARGING_SITE_1 = "charing_site_1"
    CHARGING_SITE_2 = "charing_site_2"
    CAR_SITE_1 = "car_site_1"
    CAR_SITE_2 = "car_site_2"
    CAR_SITE_DONE = "car_site_done"
    EQ_SOCKET = "eq_socgun_1"     # 枪-插座 weld
    GUN_COL_6 = "gun_col_6"

    def __init__(self):
        super().__init__()
        self.success_thresh = 0.01

    def reset(self, seed: int | None = None):
        super().reset(seed)
        # 标记关键 site 供 viewer 可视化
        names = [self.GUN_SITE, self.GUN_SITE_1, self.GUN_SITE_2,
                 self.CHARGING_SITE_1, self.CHARGING_SITE_2,
                 self.CAR_SITE_1, self.CAR_SITE_2, self.CAR_SITE_DONE]
        self.markers = []
        for n in names:
            try:
                p, _ = self.env.site_pose(n)
                self.markers.append({"pos": p.copy(),
                                     "rgba": [1, 0, 1, 1], "size": 0.012})
            except ValueError:
                pass
        return {}

    def compute_reward(self, obs: np.ndarray):
        # 序列任务的奖励由 run 脚本的阶段逻辑驱动，这里返回中性值
        return 0.0, False, {}

    # ---- 便捷：site 位姿 ----
    def site_pos(self, name: str) -> np.ndarray:
        return self.env.site_pose(name)[0]

    def site_mat(self, name: str) -> np.ndarray:
        return self.env.site_pose(name)[1]
