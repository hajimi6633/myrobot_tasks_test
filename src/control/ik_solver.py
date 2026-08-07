"""逆运动学：基于 MuJoCo 雅可比的阻尼最小二乘迭代（数值 IK）。

当目标含姿态约束时，采用两阶段求解：
  阶段1：仅位置 IK，快速逼近目标坐标
  阶段2：以阶段1解为初值，联合位置+姿态 IK 精调
这样避免从远端初值同时收敛 6 维误差时陷入局部最优或发散。

支持耦合体（GraspCoupler）：当 site_id 指向被耦合物体上的 site 时，
迭代中每次修改 arm qpos 后同步更新物体 freejoint，使该 site 的
位姿跟随臂运动。由于 freejoint 不在臂运动链中，mj_jacSite 返回的
雅可比为零，此处改用解析雅可比：通过锚点 body（gripper_base）的
雅可比 + 刚性偏移叉乘推导耦合 site 的雅可比。
"""
from __future__ import annotations
import numpy as np
import mujoco


class IKSolver:
    def __init__(self, env, damping: float = 1e-2, max_iter: int = 100,
                 tol: float = 1e-4):
        self.env = env
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.arm_dofadr = env.arm_dofadr
        self.arm_qposadr = env.arm_qposadr
        self.ee_site_id = env.ee_site_id
        # 关节限位范围 (n_arm, 2)，用于迭代中钳位防止超限
        self.jnt_range = np.array(
            [env.model.jnt_range[j] for j in env.arm_jids], dtype=float)
        # 关节权重：腕关节（后 3 个）权重更高，IK 优先用肩肘而非弱腕关节
        # （腕关节 forcerange=28 远小于肩肘的 500，大角度变化位置控制器无法跟踪）
        n_arm = len(env.arm_jids)
        self.joint_weights = np.ones(n_arm)
        self.joint_weights[3:] = 4.0  # wrist_1/2/3 权重 4x
        # 耦合器：抓取后设置，使 IK 能对被耦合物体上的 site 求解
        self.coupler = None
        self._coupler_dt = 0.05  # coupler.update 的 dt（仅用于速度估计，不影响位置）

    def _sync_coupled(self):
        """arm qpos 修改后同步更新耦合体 freejoint + mj_forward。

        被耦合物体（如充电枪）通过 freejoint 独立于臂运动链，修改 arm qpos
        后其 site_xpos 不会变化。此处手动更新 freejoint 使其跟随末端，
        然后 mj_forward 重算 site_xpos，使 IK 迭代能"看到"枪体移动。
        """
        if self.coupler is not None and self.coupler.attached:
            self.coupler.update(self._coupler_dt)
            mujoco.mj_forward(self.env.model, self.env.data)

    def _coupled_jac(self, sid: int, jacp: np.ndarray, jacr: np.ndarray):
        """解析计算耦合体上 site 的雅可比。

        被耦合物体经 GraspCoupler 刚性绑定到锚点 body（gripper_base），
        site 位姿 = 锚点位姿 + 刚性偏移。mj_jacSite 对 freejoint 体返回
        零雅可比（不在臂运动链中），故改用锚点 body 雅可比推导：
          J_pos_site = J_pos_anchor - skew(offset_world) @ J_rot_anchor
          J_rot_site = J_rot_anchor（角速度相同）
        """
        m, d = self.env.model, self.env.data
        anchor_bid = self.coupler.anchor_bid
        # 锚点 body 的雅可比
        jacp_anchor = np.zeros((3, m.nv))
        jacr_anchor = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp_anchor, jacr_anchor, anchor_bid)
        # site 相对锚点的世界偏移
        site_pos = d.site_xpos[sid]
        anchor_pos = d.xpos[anchor_bid]
        offset = site_pos - anchor_pos
        # 叉乘斜对称矩阵：ω × offset = skew(offset) @ ω
        sk = np.array([
            [0, -offset[2], offset[1]],
            [offset[2], 0, -offset[0]],
            [-offset[1], offset[0], 0],
        ])
        # 位置雅可比 = 锚点位置雅可比 - skew(offset) @ 锚点角速度雅可比
        jacp[:] = jacp_anchor - sk @ jacr_anchor
        jacr[:] = jacr_anchor

    def _solve_stage(self, target_pos, target_rot, q_init, max_iter, step_clip,
                     site_id=None, z_align_only=False):
        """单阶段 IK 迭代（内部，不备份/复原状态，由 solve 统一管理）。

        z_align_only: 仅约束 site z 轴方向与 target_rot 的 z 轴平行（不约束绕 z 轴
        自转），满足"z 轴平行"要求同时放松 1 个自由度，避免全姿态匹配在工作空间
        边缘不可解。误差用 cross(cur_z, tgt_z)（最小旋转对齐 z 轴）。
        """
        m, d = self.env.model, self.env.data
        sid = site_id if site_id is not None else self.ee_site_id
        # 耦合体上的 site 需用解析雅可比
        use_coupled_jac = (self.coupler is not None and self.coupler.attached
                           and sid != self.ee_site_id)
        if q_init is not None:
            d.qpos[self.arm_qposadr] = q_init
        mujoco.mj_forward(m, d)
        self._sync_coupled()

        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))

        for _ in range(max_iter):
            cur_pos = d.site_xpos[sid].copy()
            err_p = target_pos - cur_pos
            pos_done = np.linalg.norm(err_p) < self.tol

            rot_done = True
            err_r = np.zeros(3)
            if target_rot is not None:
                cur_rot = d.site_xmat[sid].reshape(3, 3)
                if z_align_only:
                    # z 轴对齐：误差 = cross(当前z, 目标z)，最小旋转使两 z 轴平行
                    cur_z = cur_rot[:, 2]
                    tgt_z = target_rot[:, 2]
                    err_r = np.cross(cur_z, tgt_z)
                else:
                    R_err = target_rot @ cur_rot.T
                    # 旋转误差：R_err 的反对称部分 ≈ 2*sin(θ)*axis
                    err_r = np.array([
                        R_err[2, 1] - R_err[1, 2],
                        R_err[0, 2] - R_err[2, 0],
                        R_err[1, 0] - R_err[0, 1],
                    ])
                rot_done = np.linalg.norm(err_r) < self.tol

            if pos_done and rot_done:
                break

            if use_coupled_jac:
                self._coupled_jac(sid, jacp, jacr)
            else:
                mujoco.mj_jacSite(m, d, jacp, jacr, sid)
            jac = jacp[:, self.arm_dofadr]
            if target_rot is not None:
                jac = np.vstack([jac, jacr[:, self.arm_dofadr]])
                err = np.concatenate([err_p, err_r])
            else:
                err = err_p

            # 加权阻尼最小二乘：dq = W^{-1} J^T (J W^{-1} J^T + λI)^{-1} e
            # 通过缩放雅可比列实现：J'=J·diag(1/√w), dq'=J'^T(J'J'^T+λI)^{-1}e, dq=dq'·diag(1/√w)
            # 权重大的关节（腕关节）缩放后雅可比列更小，IK 倾向用肩肘而非腕
            w_inv_sqrt = 1.0 / np.sqrt(self.joint_weights)
            jac_w = jac * w_inv_sqrt[np.newaxis, :]
            JJt = jac_w @ jac_w.T + self.damping**2 * np.eye(jac_w.shape[0])
            dq = w_inv_sqrt * (jac_w.T @ np.linalg.solve(JJt, err))
            dq = np.clip(dq, -step_clip, step_clip)
            d.qpos[self.arm_qposadr] += dq
            # 钳位到关节限位，防止 IK 解超出物理范围（如肘关节 ±π）
            d.qpos[self.arm_qposadr] = np.clip(
                d.qpos[self.arm_qposadr], self.jnt_range[:, 0],
                self.jnt_range[:, 1])
            mujoco.mj_forward(m, d)
            self._sync_coupled()

        return d.qpos[self.arm_qposadr].copy()

    def solve(self, target_pos: np.ndarray, target_rot: np.ndarray | None = None,
              q_init: np.ndarray | None = None,
              site_id: int | None = None,
              z_align_only: bool = False) -> np.ndarray:
        """解算到目标位姿，返回关节角。

        target_pos: (3,)  目标位置
        target_rot: (3,3) 目标旋转矩阵；None=只跟踪位置
        q_init: 初始关节角猜测；None=用当前仿真状态
        site_id: 目标 site 的 id；None=用 ee_link。指定后直接对该 site
                 求解位置+姿态，无需 gun↔ee 偏移转换（消除耦合误差）。
                 若该 site 在被耦合物体上（如 gun_site），需先设置 self.coupler。
        z_align_only: True=仅约束 z 轴方向平行（不约束绕 z 自转），适合工作空间
                      边缘全姿态不可解的场景。

        含姿态约束时自动两阶段求解：先位置-only 逼近，再联合精调。
        """
        m, d = self.env.model, self.env.data
        qpos_backup = d.qpos.copy()  # 备份仿真状态

        try:
            if target_rot is not None:
                # 阶段1：仅位置，大步长快速逼近
                q_stage1 = self._solve_stage(
                    target_pos, None, q_init,
                    max_iter=self.max_iter, step_clip=0.5, site_id=site_id)
                # 阶段2：位置+姿态，小步长精调
                q_sol = self._solve_stage(
                    target_pos, target_rot, q_stage1,
                    max_iter=self.max_iter, step_clip=0.2, site_id=site_id,
                    z_align_only=z_align_only)
            else:
                q_sol = self._solve_stage(
                    target_pos, None, q_init,
                    max_iter=self.max_iter, step_clip=0.3, site_id=site_id)
        finally:
            # 复原仿真状态，避免 IK 修改 data.qpos 影响仿真
            d.qpos[:] = qpos_backup
            mujoco.mj_forward(m, d)
        return q_sol
