"""约束管理：激活/停用 equality 约束（如充电枪 weld）。

MuJoCo 3.x 中运行时约束激活状态存储在 data.eq_active（bool 数组），
model.eq_active0 仅为 mj_resetData 时的初始值。故此处操作 data.eq_active。
"""
from __future__ import annotations
import numpy as np
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

    def set_weld_relpose(self, name: str, body1: str, body2: str):
        """把 body2 相对 body1 的当前位姿写入 weld 的 eq_data。

        激活 weld 前必须调用：eq_data 存的相对位姿是编译时值（本模型中
        枪在插座、臂在 home 的构型），与激活时刻的真实相对位姿可相差
        1m 以上，直接激活会把两 body 瞬间猛拉到编译位姿——巨力跳变。
        写入当前值后激活，约束残差为零，无任何瞬态。

        eq_data 布局（weld，共 11 元素）：
        [anchor(3), relpos(3), relquat(4), torquescale(1)]
        相对位姿定义：p2 = p1 + R(q1) @ relpos，q2 = q1 ⊗ relquat
        （relpos 为 body1 局部系坐标，与 GraspCoupler.attach 的记录一致）
        """
        e = self._eq_id(name)
        m, d = self.model, self.data
        b1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body1)
        b2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body2)
        if b1 < 0 or b2 < 0:
            raise ValueError(f"body not found: {body1} / {body2}")
        q1, q2 = d.xquat[b1], d.xquat[b2]
        q1_inv = np.empty(4)
        mujoco.mju_negQuat(q1_inv, q1)  # 单位四元数的逆 = 共轭
        rel_quat = np.empty(4)
        mujoco.mju_mulQuat(rel_quat, q1_inv, q2)
        rel_pos = np.empty(3)
        # 世界系偏移旋回 body1 局部系
        mujoco.mju_rotVecQuat(rel_pos, d.xpos[b2] - d.xpos[b1], q1_inv)
        m.eq_data[e, 3:6] = rel_pos
        m.eq_data[e, 6:10] = rel_quat
