"""抓取耦合记录：供 IK 数值迭代时虚拟同步被夹物体。

物理固定由 eq_gun_ee weld 完成（激活前把当前相对位姿写入 eq_data，
见 ConstraintManager.set_weld_relpose——直接用编译时位姿激活会跳变）。
weld 在约束求解器内与接触力、执行器力同循环解算：插枪阻力经 weld
传回臂关节，穿模深度由接触力与约束力的平衡限定，无需每步代码干预。

本类职责：attach() 记录物体相对锚点 body 的位姿偏移；IK 迭代中修改
臂 qpos 后由 update() 把物体 freejoint 同步到"锚点位姿 × 偏移"的虚拟
位姿（纯运动学、求解结束由 IKSolver 整体复原），使雅可比与收敛判据
能"看到"物体上的 site 跟随臂运动。物体始终参与碰撞检测，接触力可被
force/torque sensor 读到（供导纳控制）。
"""
from __future__ import annotations
import numpy as np
import mujoco


class GraspCoupler:
    """记录 freejoint 物体相对末端 body 的位姿（供 IK 虚拟同步）。"""

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
        """把 object freejoint qpos 设为 anchor 当前位姿 × 记录偏移。

        供 IK 迭代中的虚拟同步（_sync_coupled）与个别一次性传送使用。
        dt>0 时用差分估计 qvel 保持一致；dt=0 仅同步位姿不写 qvel——
        weld 激活后的一次性传送必须用 0，避免向物理状态注入传送速度。
        """
        if not self.attached:
            return
        p_anc = self.data.xpos[self.anchor_bid].copy()
        q_anc = self.data.xquat[self.anchor_bid].copy()
        p_obj = p_anc + _quat_rot(q_anc, self.rel_pos)
        q_obj = _quat_mul(q_anc, self.rel_quat)
        new_qpos = np.concatenate([p_obj, q_obj])
        # 差分速度（含自适应限幅）：_prev_qpos 若被外部调用方（如 IK 迭代）
        # 留在陈旧位姿，差分会算出巨大速度写入 qvel，物体在下一步物理仿真
        # 中被高速"发射"刮碰末端（曾实测 9~21m/s）。上限取锚点当前速度的
        # 物理合法值（v + ω×偏移）再加裕量——固定小阈值会截断大跨度快速
        # 段的合法跟随速度（臂端峰值可达 2~3m/s）
        if dt > 0:
            v = (p_obj - self._prev_qpos[:3]) / dt
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data,
                                      mujoco.mjtObj.mjOBJ_BODY,
                                      self.anchor_bid, vel6, 0)
            v_cap = (np.linalg.norm(vel6[:3])
                     + np.linalg.norm(vel6[3:]) * np.linalg.norm(p_obj - p_anc)
                     + 0.5)  # 裕量 (m/s)
            v_norm = np.linalg.norm(v)
            if v_norm > v_cap:
                v = v * (v_cap / v_norm)
            # 角速度简化：忽略，设 0（对接触力读取无影响）
            self.data.qvel[self.obj_dofadr:self.obj_dofadr + 3] = v
            self.data.qvel[self.obj_dofadr + 3:self.obj_dofadr + 6] = 0.0
        self.data.qpos[self.obj_qposadr:self.obj_qposadr + 3] = p_obj
        self.data.qpos[self.obj_qposadr + 3:self.obj_qposadr + 7] = q_obj
        self._prev_qpos = new_qpos

    def resync(self):
        """用物体当前实际 qpos 重置差分基准 _prev_qpos。

        IK 求解会经 update() 迭代物体虚拟位姿，结束时 _prev_qpos 停在
        "虚拟位姿"（距真实位姿可达 0.3~1m）。若不重置，真实位姿恢复后
        的首次 update() 会算出巨大差分速度。调用方在还原仿真状态后应
        调用本方法（如 IKSolver.solve 的 finally 段）。
        """
        self._prev_qpos = self.data.qpos[
            self.obj_qposadr:self.obj_qposadr + 7].copy()

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
