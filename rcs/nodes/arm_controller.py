"""ArmController：控制环安全闸门（50Hz，主循环）。

每拍顺序（顺序即安全语义，对话第 6 轮快通道设计）：
  1. watchdog：/safety_state 超时或 STOP → 冻结（不发布 = 维持上拍 ctrl）
  2. SLOW 区 → 指令朝当前关节角收缩（位置控制器下的等效限速）
  3. 发布 /joint_cmd
安全链直达本节点，不经过任务层——对应真机安全继电器的位置。
"""
from __future__ import annotations
import numpy as np

from rclike import Node

ZONE_NORMAL, ZONE_SLOW, ZONE_STOP = "normal", "slow", "stop"


class ArmController(Node):
    def __init__(self, bus, clock):
        super().__init__("arm_controller", bus, clock)
        self.declare_parameter("watchdog_s", 0.15, "安全状态超时即冻结 (s)")
        self.declare_parameter("slow_scale", 0.25, "减速区指令收缩比例")
        self.declare_parameter("require_safety", False,
                               "True 时无安全数据视为不安全（真机部署开启）")
        self._safety = self.bus.topic("/safety_state")
        self._states = self.bus.topic("/joint_states")
        self._pub = self.create_publisher("/joint_cmd")
        self._pending = None          # (q_arm, grip)：上游发布的期望指令
        self._frozen = False
        self.create_subscription("/q_des", self._on_qdes)

    def _on_qdes(self, msg, stamp):
        """只存引用立即返回（上游可为任务层或伺服节点）。"""
        self._pending = msg

    def on_tick(self):
        if self._zone() == ZONE_STOP or self._watchdog_expired():
            if not self._frozen:
                self.log.warn("安全停止：冻结关节指令（夹爪与 weld 状态保持）")
                self._frozen = True
            return                       # 不发布 = SimNode 维持上一拍 ctrl
        self._frozen = False
        if self._pending is None:
            return
        q_des, grip = self._pending
        if self._zone() == ZONE_SLOW:
            q_des = self._scale(q_des)
        self._pub.publish((q_des, grip), stamp=self.clock.now)

    # ---- 安全读取 ----
    def _zone(self) -> str:
        s = self._safety.latest
        if s is None:
            # 无安全数据：仿真默认放行；真机（require_safety=True）视为失明
            return ZONE_STOP if self.get_parameter("require_safety") else ZONE_NORMAL
        return s[0].get("zone", ZONE_STOP)

    def _watchdog_expired(self) -> bool:
        s = self._safety.latest
        return s is not None and \
            (self.clock.now - s[1]) > self.get_parameter("watchdog_s")

    def _scale(self, q_des) -> np.ndarray:
        """减速区：目标朝当前关节角收缩（q_cur + s·(q_des−q_cur)）。"""
        st = self._states.latest
        if st is None:
            return q_des
        q_cur = st[0][0]                # /joint_states msg = (qpos, qvel)
        return q_cur + self.get_parameter("slow_scale") * \
            (np.asarray(q_des, float) - q_cur)
