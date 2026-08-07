"""抓取耦合：抓取后用运动学方式让物体跟随末端 body。

MuJoCo weld 约束动态激活时会用编译时相对位姿（枪在插座处）而非抓取瞬间位姿，
导致跳变。这里改为运动学耦合：attach 时记录物体相对末端的位姿偏移，
之后每步把物体 freejoint 的 qpos/qvel 设为末端当前位姿 × 偏移。
物体仍参与碰撞检测，接触力可被 force/torque sensor 读到（供导纳控制）。
"""
from __future__ import annotations
import numpy as np
import mujoco


class GraspCoupler:
    """把一个 freejoint 物体运动学绑定到某个末端 body。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 object_body: str, anchor_body: str):
        self.model = model
        self.data = data
        self.obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
        self.anchor_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, anchor_body)
        # object freejoint 的 qpos/qvel 地址
        jid = int(model.body_jntadr[self.obj_bid])
        self.obj_qposadr = int(model.jnt_qposadr[jid])
        self.obj_dofadr = int(model.jnt_dofadr[jid])
        self.attached = False
        self.rel_pos = np.zeros(3)        # object 相对 anchor（anchor local）
        self.rel_quat = np.array([1.0, 0, 0, 0])
        self._prev_qpos = np.zeros(7)

    def attach(self):
        """记录当前 object 相对 anchor 的位姿，开启跟随。"""
        p_obj = self.data.xpos[self.obj_bid].copy()
        q_obj = self.data.xquat[self.obj_bid].copy()
        p_anc = self.data.xpos[self.anchor_bid].copy()
        q_anc = self.data.xquat[self.anchor_bid].copy()
        q_anc_inv = _quat_inv(q_anc)
        dp = p_obj - p_anc
        self.rel_pos = _quat_rot(q_anc_inv, dp)
        self.rel_quat = _quat_mul(q_anc_inv, q_obj)
        self._prev_qpos = np.concatenate([p_obj, q_obj])
        self.attached = True

    def update(self, dt: float):
        """每步调用：把 object freejoint qpos 设为 anchor 当前位姿 × 记录偏移，
        并用差分估计 qvel 保持一致。"""
        if not self.attached:
            return
        p_anc = self.data.xpos[self.anchor_bid].copy()
        q_anc = self.data.xquat[self.anchor_bid].copy()
        p_obj = p_anc + _quat_rot(q_anc, self.rel_pos)
        q_obj = _quat_mul(q_anc, self.rel_quat)
        new_qpos = np.concatenate([p_obj, q_obj])
        # 差分速度
        if dt > 0:
            v = (p_obj - self._prev_qpos[:3]) / dt
            # 角速度简化：忽略，设 0（对接触力读取无影响）
            self.data.qvel[self.obj_dofadr:self.obj_dofadr + 3] = v
            self.data.qvel[self.obj_dofadr + 3:self.obj_dofadr + 6] = 0.0
        self.data.qpos[self.obj_qposadr:self.obj_qposadr + 3] = p_obj
        self.data.qpos[self.obj_qposadr + 3:self.obj_qposadr + 7] = q_obj
        self._prev_qpos = new_qpos

    def detach(self):
        self.attached = False


# ---- 四元数工具（w,x,y,z）----
def _quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mul(q0, q1):
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return np.array([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ])


def _quat_rot(q, v):
    """四元数 q 旋转向量 v。"""
    w, x, y, z = q
    qvec = np.array([x, y, z])
    return v + 2.0 * np.cross(qvec, np.cross(qvec, v) + w * v)
