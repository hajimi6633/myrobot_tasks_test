"""约束管理：激活/停用 equality 约束（如充电枪 weld）。

MuJoCo 3.x 中运行时约束激活状态存储在 data.eq_active（bool 数组），
model.eq_active0 仅为 mj_resetData 时的初始值。故此处操作 data.eq_active。
"""
from __future__ import annotations
import mujoco


class ConstraintManager:
    """封装 data.eq_active 的读写，按约束名控制运行时激活状态。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

    def _eq_id(self, name: str) -> int:
        eid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eid < 0:
            raise ValueError(f"equality '{name}' not found")
        return eid

    def set_active(self, name: str, active: bool):
        """激活或停用指定 equality 约束（运行时，修改 data.eq_active）。"""
        self.data.eq_active[self._eq_id(name)] = bool(active)

    def is_active(self, name: str) -> bool:
        return bool(self.data.eq_active[self._eq_id(name)])
