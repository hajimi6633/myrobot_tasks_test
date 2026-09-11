"""IKService：把现有 IKSolver 包装成服务节点。

绞杀式迁移的关键一步（对话第 1 轮）：SimNode 的属性命名与旧
ArmEnv 保持一致，因此 src/control/ik_solver.py 的 IKSolver
不改一行即可从 IKSolver(env) 换绑为 IKSolver(sim)。
"""
from __future__ import annotations

from rclike import Node
from src.control.ik_solver import IKSolver       # 直接复用旧求解器


class IKService(Node):
    def __init__(self, bus, clock, sim):
        super().__init__("ik_service", bus, clock)
        self.ik = IKSolver(sim)                   # env → sim 无缝换绑
        self.create_service("solve_ik", self._solve)

    def _solve(self, req: dict):
        """req: {target_pos, target_rot?, q_init?, site_id?,
        z_align_only?, retry?, max_travel?} —— 与旧 solve 接口一致。"""
        return self.ik.solve(**req)
