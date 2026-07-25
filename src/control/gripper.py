"""末端执行器 / 夹爪控制：开合 + 接触力读取。"""
from __future__ import annotations
import numpy as np
import mujoco


class Gripper:
    """二指平行夹爪封装。

    action 维度=1：[0,1]，0=完全闭合，1=完全张开。
    内部映射到 left_joint / right_joint 两个 slide joint。
    joint 名与 body 名均由外部传入（来自 config/gripper.yaml），更换模型无需改代码。
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 left_joint: str = "ll_grasp_joint", right_joint: str = "rl_grasp_joint",
                 left_body: str = "finger_left", right_body: str = "finger_right"):
        self.model = model
        self.data = data
        self.left_joint = left_joint
        self.right_joint = right_joint
        # 接触力检测用的手指 body 名集合
        self.finger_bodies = (left_body, right_body)
        self.jid_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, left_joint)
        self.jid_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, right_joint)
        self.qposadr_l = model.jnt_qposadr[self.jid_l]
        self.qposadr_r = model.jnt_qposadr[self.jid_r]
        self.dofadr_l = model.jnt_dofadr[self.jid_l]
        self.dofadr_r = model.jnt_dofadr[self.jid_r]
        # 行程范围（取自 joint range，model 为真值）
        self.range_l = model.jnt_range[self.jid_l]
        self.range_r = model.jnt_range[self.jid_r]

    def set_target(self, open_ratio: float):
        """open_ratio in [0,1]: 0=close, 1=open.

        几何：ll_grasp range [-0.025, 0.009] qpos 增大=朝中心(闭合)，qpos 减小=外扩(张开)；
        rl_grasp range [-0.009, 0.025] 由 equality 镜像 qpos_r = -qpos_l。
        故 open=1 → qpos_l=range_l[0]=-0.025(张开)，open=0 → qpos_l=range_l[1]=0.009(闭合)。
        """
        open_ratio = float(np.clip(open_ratio, 0.0, 1.0))
        target_l = self.range_l[1] - open_ratio * (self.range_l[1] - self.range_l[0])
        target_r = self.range_r[0] - open_ratio * (self.range_r[0] - self.range_r[1])
        return target_l, target_r

    def get_opening(self) -> float:
        """返回当前开合比例 [0,1]，1=张开, 0=闭合。"""
        cur_l = self.data.qpos[self.qposadr_l]
        span = self.range_l[1] - self.range_l[0]
        if span == 0:
            return 0.0
        return 1.0 - float((cur_l - self.range_l[0]) / span)

    def get_contact_force(self) -> float:
        """两指间接触法向力之和（粗略，用于判断是否夹住物体）。"""
        force = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            # 一方为手指 geom 则累加法向力
            if self._is_finger_body(g1) or self._is_finger_body(g2):
                f = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, f)
                force += abs(f[0])
        return float(force)

    def _is_finger_body(self, body_id: int) -> bool:
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        return name in self.finger_bodies
