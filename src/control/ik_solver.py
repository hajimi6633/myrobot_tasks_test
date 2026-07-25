"""逆运动学：基于 MuJoCo 雅可比的阻尼最小二乘迭代（数值 IK）。"""
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

    def solve(self, target_pos: np.ndarray, target_rot: np.ndarray | None = None,
              q_init: np.ndarray | None = None) -> np.ndarray:
        """解算到目标位姿，返回关节角。只优化位置（旋转可选）。

        target_pos: (3,)  target_rot: (3,3) 旋转矩阵，None=只跟踪位置。
        """
        m, d = self.env.model, self.env.data
        # 备份仿真状态，求解完成后复原，避免 IK 修改 data.qpos 影响仿真
        qpos_backup = d.qpos.copy()
        if q_init is not None:
            d.qpos[self.arm_qposadr] = q_init
        mujoco.mj_forward(m, d)

        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))

        for _ in range(self.max_iter):
            cur_pos = d.site_xpos[self.ee_site_id].copy()
            err_p = target_pos - cur_pos
            pos_done = np.linalg.norm(err_p) < self.tol

            rot_done = True
            err_r = np.zeros(3)
            if target_rot is not None:
                cur_rot = d.site_xmat[self.ee_site_id].reshape(3, 3)
                R_err = target_rot @ cur_rot.T
                err_r = np.array([
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1],
                ])
                rot_done = np.linalg.norm(err_r) < self.tol

            if pos_done and rot_done:
                break

            mujoco.mj_jacSite(m, d, jacp, jacr, self.ee_site_id)
            jac = jacp[:, self.arm_dofadr]
            if target_rot is not None:
                jac = np.vstack([jac, jacr[:, self.arm_dofadr]])
                err = np.concatenate([err_p, err_r])
            else:
                err = err_p

            # 阻尼最小二乘：dq = J^T (J J^T + λI)^-1 e
            JJt = jac @ jac.T + self.damping**2 * np.eye(jac.shape[0])
            dq = jac.T @ np.linalg.solve(JJt, err)
            dq = np.clip(dq, -0.3, 0.3)  # 单步限幅，避免发散
            d.qpos[self.arm_qposadr] += dq
            mujoco.mj_forward(m, d)

        # 保存 IK 解，复原仿真状态
        q_sol = d.qpos[self.arm_qposadr].copy()
        d.qpos[:] = qpos_backup
        mujoco.mj_forward(m, d)
        return q_sol
