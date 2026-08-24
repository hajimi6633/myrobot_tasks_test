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
                 tol: float = 1e-4, rot_tol: float = 0.01):
        self.env = env
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        # 姿态收敛阈值必须独立于位置 tol：1e-4 rad ≈ 0.006deg 物理上
        # 达不到，姿态欠收敛时 100 次迭代全部跑满，关节在"位置已收敛、
        # 姿态追不动"状态下持续漂移（零空间行程），导致 q_des 单步跳变
        self.rot_tol = float(rot_tol)  # ≈0.57deg，闭环控制足够
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
        # 关节权重：默认腕关节 4x（位置主导时抑制腕部大幅动作）；
        # 姿态约束时腕部是调整姿态的自然关节（绕末端 z 自转靠 wrist_3），
        # 恢复 1.0，否则 90° 姿态旋转会被挤到肩肘造成大构型重构
        weights = self.joint_weights.copy()
        if target_rot is not None:
            weights[3:] = 1.0
        w_inv_sqrt = 1.0 / np.sqrt(weights)
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
                    # z 轴对齐：用最短旋转对齐两 z 轴（axis-angle 形式）
                    # 注意：不能用 cross(cur_z, tgt_z) 作为误差，因为当两 z 轴
                    # 反向时 cross=0 会误判收敛。正确做法是 axis*angle：
                    #   axis = normalize(cross(cur_z, tgt_z))
                    #   angle = atan2(|cross|, dot)
                    # 反向时 angle=π，IK 会正确地把姿态拉回同向。
                    cur_z = cur_rot[:, 2]
                    tgt_z = target_rot[:, 2]
                    cross_z = np.cross(cur_z, tgt_z)
                    n_cross = np.linalg.norm(cross_z)
                    dot_z = float(np.dot(cur_z, tgt_z))
                    if n_cross > 1e-9:
                        axis = cross_z / n_cross
                        angle = float(np.arctan2(n_cross, dot_z))
                        err_r = axis * angle
                    else:
                        # cross≈0：同向(dot>0，已对齐)或反向(dot<0，需绕任意正交轴转π)
                        if dot_z < 0:
                            # 反向：绕 cur_z 的任一正交轴转 π
                            # 选与 cur_z 最不正交的世界轴叉乘，得到正交轴
                            ref = np.array([1.0, 0, 0]) if abs(cur_z[0]) < 0.9 \
                                else np.array([0, 1.0, 0])
                            axis = np.cross(cur_z, ref)
                            axis = axis / np.linalg.norm(axis)
                            err_r = axis * np.pi
                        else:
                            err_r = np.zeros(3)
                else:
                    R_err = target_rot @ cur_rot.T
                    # 旋转误差：R_err 的反对称部分 ≈ 2*sin(θ)*axis
                    err_r = np.array([
                        R_err[2, 1] - R_err[1, 2],
                        R_err[0, 2] - R_err[2, 0],
                        R_err[1, 0] - R_err[0, 1],
                    ])
                rot_done = np.linalg.norm(err_r) < self.rot_tol

            if pos_done and rot_done:
                # 调试：break 时打印真实误差（暴露"伪收敛"）
                if np.linalg.norm(err_p) > 1e-3:
                    print(f"  [IK伪收敛?] break err_p={np.linalg.norm(err_p):.6f}")
                break

            # 调试：超过 max_iter 仍未收敛时，最后一轮的状态
            if _ == max_iter - 1:
                final_err = np.linalg.norm(err_p)
                print(f"  [IK未收敛] iter={max_iter} 位置err={final_err:.6f} "
                      f"site_xpos={np.round(d.site_xpos[sid],4)} "
                      f"target={np.round(target_pos,4)}")

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
              z_align_only: bool = False,
              retry: bool = True,
              max_travel: float | None = None) -> np.ndarray:
        """解算到目标位姿，返回关节角。

        target_pos: (3,)  目标位置
        target_rot: (3,3) 目标旋转矩阵；None=只跟踪位置
        q_init: 初始关节角猜测；None=用当前仿真状态
        site_id: 目标 site 的 id；None=用 ee_link。指定后直接对该 site
                 求解位置+姿态，无需 gun↔ee 偏移转换（消除耦合误差）。
                 若该 site 在被耦合物体上（如 gun_site），需先设置 self.coupler。
        z_align_only: True=仅约束 z 轴方向平行（不约束绕 z 自转），适合工作空间
                      边缘全姿态不可解的场景。
        retry: True=解后验证不达标时多起点重试（大跨度运动段用）；
               False=单次求解直接返回（闭环实时控制段用，如导纳插枪——
               目标每步只移动 ~2mm，本步欠收敛的误差下一步闭环自纠，
               重试跳出局部极小反而会造成 q_des 单步跳变/关节突变）。
        max_travel: 限制解相对 q_init 的最大关节行程 (rad)。闭环段设置后，
               即使 IK 姿态欠收敛产生零空间漂移，输出也被硬性截断在
               q_init 附近（超限时沿解方向等比缩放），保证 q_des 单步
               变化有界——截断产生的欠收敛由下一步闭环自纠。

        含姿态约束时自动两阶段求解：先位置-only 逼近，再联合精调。
        解后独立验证：若位置误差超过阈值（1mm），从扰动初值自动重试
        （最多 N 次），取误差最小的解。用于对抗迭代震荡 / 局部极小。
        """
        m, d = self.env.model, self.env.data
        qpos_backup = d.qpos.copy()  # 备份仿真状态
        rng = np.random.default_rng(0)  # 固定种子保证可重复
        VERIFY_TOL = 1e-3    # 位置验证阈值 (m)
        ROT_TOL_DEG = 2.0    # 姿态验证阈值 (deg)
        MAX_RETRY = 6        # 最多尝试次数（含首次）
        PERTURB = 0.25       # 重试初值扰动幅度 (rad)
        TRAVEL_W = 0.005     # 行程惩罚权重 (m/rad)：偏好离参考构型近的解

        def _attempt(q0):
            """一次完整求解（两阶段或单阶段），返回 q_sol。"""
            if target_rot is not None:
                # 阶段1：仅位置，大步长快速逼近
                q_stage1 = self._solve_stage(
                    target_pos, None, q0,
                    max_iter=self.max_iter, step_clip=0.5, site_id=site_id)
                # 阶段2：位置+姿态，小步长精调
                return self._solve_stage(
                    target_pos, target_rot, q_stage1,
                    max_iter=self.max_iter, step_clip=0.2, site_id=site_id,
                    z_align_only=z_align_only)
            return self._solve_stage(
                target_pos, None, q0,
                max_iter=self.max_iter, step_clip=0.3, site_id=site_id)

        def _verify(q):
            """独立验证：位置误差 (m) + 姿态误差 (deg)。"""
            sid = site_id if site_id is not None else self.ee_site_id
            d.qpos[self.arm_qposadr] = q
            mujoco.mj_forward(m, d)
            self._sync_coupled()
            pos_err = float(np.linalg.norm(target_pos - d.site_xpos[sid]))
            rot_deg = 0.0
            if target_rot is not None:
                R = d.site_xmat[sid].reshape(3, 3)
                if z_align_only:
                    # 验证口径必须与约束一致：z_align_only 只约束 z 轴平行，
                    # 若此处用全姿态转角，绕 z 自转分量会被误报为姿态误差
                    cos_z = float(np.clip(np.dot(R[:, 2], target_rot[:, 2]),
                                          -1.0, 1.0))
                    rot_deg = float(np.degrees(np.arccos(abs(cos_z))))
                else:
                    R_e = target_rot @ R.T
                    cos_t = float(np.clip((np.trace(R_e) - 1.0) / 2.0, -1.0, 1.0))
                    rot_deg = float(np.degrees(np.arccos(abs(cos_t))))
            return pos_err, rot_deg

        # 行程参考：q_init 或求解时刻的臂构型（q_init 为 None 时用后者）
        q_ref = (q_init if q_init is not None
                 else qpos_backup[self.arm_qposadr].copy())

        try:
            n_attempts = MAX_RETRY if retry else 1
            best_q, best_score, best_info = None, np.inf, None
            q0 = q_init
            for attempt in range(n_attempts):
                q_sol = _attempt(q0)
                pos_err, rot_deg = _verify(q_sol)
                travel = float(np.max(np.abs(q_sol - q_ref)))
                # 综合评分：位置 + 姿态 + 行程惩罚（同精度下偏好小行程解，
                # 避免构型翻转导致控制器跟踪不上）
                score = pos_err + 0.002 * rot_deg + TRAVEL_W * travel
                if score < best_score:
                    best_q, best_score = q_sol.copy(), score
                    best_info = (pos_err, rot_deg, travel)
                if pos_err < VERIFY_TOL and rot_deg < ROT_TOL_DEG:
                    break
                # 未达阈值：在当前最优解附近扰动重试（盆地内探索）。
                # 仅在确实还有下一次尝试时执行/打印——retry=False 时
                # 循环只跑 1 轮，此处再打印"[IK重试]"是误导日志
                if attempt + 1 < n_attempts:
                    q0 = np.clip(best_q + rng.normal(0.0, PERTURB, size=len(best_q)),
                                 self.jnt_range[:, 0], self.jnt_range[:, 1])
                    print(f"  [IK重试] attempt={attempt+1} pos={pos_err:.4f}m "
                          f"rot={rot_deg:.1f}deg travel={travel:.2f}rad, "
                          f"在最优解附近扰动重试")
            if best_info[0] >= VERIFY_TOL or best_info[1] >= ROT_TOL_DEG:
                tag = "" if retry else " (retry=False, 闭环段欠收敛由下一步自纠)"
                print(f"  [IK警告] 最优解 pos={best_info[0]:.4f}m "
                      f"rot={best_info[1]:.1f}deg travel={best_info[2]:.2f}rad "
                      f"(目标={np.round(target_pos,4)}){tag}")
            q_sol = best_q
        finally:
            # 复原仿真状态，避免 IK 修改 data.qpos 影响仿真
            d.qpos[:] = qpos_backup
            mujoco.mj_forward(m, d)
        # 行程硬限制：姿态欠收敛时 IK 的零空间漂移会让解离 q_init 很远
        # （曾出现 0.76rad/步），此处沿解方向等比缩放截断到 max_travel，
        # 保证闭环段 q_des 单步变化有界；截断欠收敛由下一步闭环自纠
        if max_travel is not None:
            travel = float(np.max(np.abs(q_sol - q_ref)))
            if travel > max_travel:
                q_sol = q_ref + (q_sol - q_ref) * (max_travel / travel)
        return q_sol
