"""充电枪任务 6 阶段构建：每个子段为一个独立 Phase，解耦可插拔。

执行约定（run 脚本遵循）：
  - 每个 Phase 的 trajectory=None，由 on_step(ctx, i) 返回 q_des
  - on_enter 在阶段开始时规划轨迹（此时 site 位姿已正确），存入闭包 st
  - 导纳段 on_step 实时 IK（读力→导纳修正→求解）
  - 每步后检查 done_condition，True 则提前结束
  - 阶段切换调 on_enter / on_exit
"""
from __future__ import annotations
import numpy as np
import mujoco

from src.control.trajectory import min_jerk, Trajectory
from src.env.tasks.charging_gun import Phase, PhaseContext, ChargingGunTask as Task

# ---- 参数 ----
DT = 0.05
HOLD_1S = int(1.0 / DT)
GRASP_FORCE_MIN = 1.0      # 抓取成功最小夹爪力 (N)
F_BLOCK = 5.0              # 插枪阻力阈值 (N)，超过则暂停主推进
INSERT_STEP = 0.002        # 插枪名义推进步长 (m)
INSERT_MAX_STEPS = 400     # 插枪段最大步数
DONE_DIST = 0.006          # 到底距离阈值 (m)
SEARCH_STEP = 0.0008       # 受阻时横向搜索步长 (m)，朝目标 xy 对齐寻找突破口
# 归位段路径短且易撞插座外壳，用更慢步长 + 更低力阈值
INSERT_STEP_SLOW = 0.0006  # 归位段慢速推进步长 (m)
F_BLOCK_SLOW = 2.0         # 归位段阻力阈值 (N)
GRIP_CLOSE = 0.0
GRIP_OPEN = 1.0


# ---- 辅助 ----
def _plan(ctx: PhaseContext, target_pos, target_rot, T, q_init=None,
          site_id=None, z_align_only=False):
    """规划 min_jerk 轨迹。site_id 指定时直接对该 site 求 IK（如 gun_site）。

    z_align_only: 仅约束 z 轴方向平行，避免全姿态匹配在工作空间边缘不可解。
    """
    q_cur = (q_init if q_init is not None
             else ctx.env.data.qpos[ctx.env.arm_qposadr].copy())
    q_goal = ctx.ik.solve(target_pos, target_rot=target_rot, q_init=q_cur,
                          site_id=site_id, z_align_only=z_align_only)
    wp = min_jerk(q_cur, q_goal, T, ctx.dt_ctrl)
    return Trajectory(wp, ctx.dt_ctrl), q_goal


def _hold_traj(q, n, dt):
    return Trajectory(np.tile(np.asarray(q, float), (n, 1)), dt)


def _gun_to_ee(ctx: PhaseContext, gun_world: np.ndarray) -> np.ndarray:
    """gun_site 世界目标 → ee_link 世界目标。

    用抓取时记录的 ee_link 局部系偏移（恒定），乘以当前 ee_link 旋转矩阵
    转回世界系。枪体经 GraspCoupler 刚性耦合到 gripper_base，ee_link 与
    gripper_base 刚性连接，故局部偏移不随姿态变化；但世界系偏移会随姿态
    改变，必须用当前旋转矩阵实时转换。
    """
    ee_mat = ctx.env.data.site_xmat[ctx.env.ee_site_id].reshape(3, 3)
    return np.asarray(gun_world, float) - ee_mat @ ctx.ee_to_gun_offset


def _gun_rot_to_ee(ctx: PhaseContext, gun_target_rot: np.ndarray) -> np.ndarray:
    """gun_site 目标旋转 → ee_link 目标旋转。

    抓取后枪与末端刚性耦合，相对旋转 R_gun_ee 固定：R_gun = R_gun_ee @ R_ee。
    故 R_ee = R_gun_ee.T @ R_gun_target，确保枪体 z 轴与目标 site z 轴平行。
    """
    return ctx.grasp_rot.T @ np.asarray(gun_target_rot, float)


def _ctrl_dt(ctx: PhaseContext) -> float:
    return ctx.env.n_substeps * ctx.env.model.opt.timestep


def _move_phase(name, desc, site_name, grip, T, offset_gun=False,
                target_rot_site=None,
                on_enter_extra=None, on_exit=None, monitor_force=False):
    """生成运动到指定 site 的 Phase。轨迹在 on_enter 规划。

    target_rot_site: 若指定，终点 IK 同时约束姿态——将枪体 z 轴对齐该 site
    的 z 轴。min_jerk 在关节空间插值，姿态平滑过渡，避免插枪段突变。
    """
    st = {"traj": None}

    def enter(c):
        mujoco.mj_forward(c.env.model, c.env.data)
        q0 = c.env.data.qpos[c.env.arm_qposadr].copy()
        pos, _ = c.env.site_pose(site_name)
        # offset_gun=True 时直接对 gun_site 求 IK（利用枪体延伸臂展），
        # target_rot 直接用目标 site 旋转矩阵（IK 匹配 gun_site 姿态）
        sid = c.gun_site_id if offset_gun else None
        target_rot = None
        z_align = False
        if target_rot_site is not None:
            _, site_mat = c.env.site_pose(target_rot_site)
            target_rot = site_mat if offset_gun else \
                _gun_rot_to_ee(c, site_mat)
            # 仅约束 z 轴平行（不约束绕 z 自转），避免全姿态在工作空间边缘不可解
            z_align = True
        target = pos  # gun_site 目标 = site 世界位置
        traj, q_goal = _plan(c, target, target_rot, T, q_init=q0,
                             site_id=sid, z_align_only=z_align)
        # 诊断：关节角度差异（检测构型翻转）
        dq_max = np.max(np.abs(q_goal - q0))
        print(f"  [JNT] {name}: q0={np.round(q0,3)} q_goal={np.round(q_goal,3)} "
              f"max|Δq|={dq_max:.3f}rad")
        # 诊断：检查 IK 解的实际 site 位置是否收敛
        if target_rot is not None or offset_gun:
            backup = c.env.data.qpos.copy()
            c.env.data.qpos[c.env.arm_qposadr] = q_goal
            mujoco.mj_forward(c.env.model, c.env.data)
            if c.coupler.attached:
                c.coupler.update(c.dt_ctrl)
                mujoco.mj_forward(c.env.model, c.env.data)
            check_sid = c.gun_site_id if offset_gun else c.env.ee_site_id
            actual = c.env.data.site_xpos[check_sid].copy()
            err = np.linalg.norm(target - actual)
            print(f"  [IK] {name}: 误差={err:.4f}m 目标={np.round(target,4)} "
                  f"实际={np.round(actual,4)}")
            c.env.data.qpos[:] = backup
            mujoco.mj_forward(c.env.model, c.env.data)
            if c.coupler.attached:
                c.coupler.update(c.dt_ctrl)
                mujoco.mj_forward(c.env.model, c.env.data)
        st["traj"] = traj
        if on_enter_extra:
            on_enter_extra(c)

    def step(c, i):
        if st["traj"] is None:
            return c.env.data.qpos[c.env.arm_qposadr].copy()
        return st["traj"].at(i)

    return Phase(name=name, desc=desc, trajectory=None, n_steps=int(T / DT),
                 grip_ratio=grip, on_enter=enter, on_step=step,
                 on_exit=on_exit, monitor_force=monitor_force)


def _move_pos_phase(name, desc, target_pos, grip, T, offset_gun=True):
    """运动到指定世界坐标（非 site）的 Phase。用于 via point 过渡。

    与 _move_phase 类似但接受位置数组而非 site 名，纯位置 IK（不约束姿态）。
    """
    st = {"traj": None}
    target = np.asarray(target_pos, float)

    def enter(c):
        mujoco.mj_forward(c.env.model, c.env.data)
        q0 = c.env.data.qpos[c.env.arm_qposadr].copy()
        sid = c.gun_site_id if offset_gun else None
        traj, q_goal = _plan(c, target, None, T, q_init=q0, site_id=sid)
        st["traj"] = traj
        dq_max = np.max(np.abs(q_goal - q0))
        print(f"  [JNT] {name}: max|Δq|={dq_max:.3f}rad via={np.round(target,3)}")

    def step(c, i):
        if st["traj"] is None:
            return c.env.data.qpos[c.env.arm_qposadr].copy()
        return st["traj"].at(i)

    return Phase(name=name, desc=desc, trajectory=None, n_steps=int(T / DT),
                 grip_ratio=grip, on_enter=enter, on_step=step)


def _hold_phase(name, desc, site_name, grip, hold_s, offset_gun=False,
                on_exit=None, monitor_force=False):
    """保持当前位姿（在指定 site 处）的 hold 段。"""
    st = {"traj": None}

    def enter(c):
        q = c.env.data.qpos[c.env.arm_qposadr].copy()
        st["traj"] = _hold_traj(q, int(hold_s / DT), c.dt_ctrl)

    def step(c, i):
        return st["traj"].at(i)

    return Phase(name=name, desc=desc, trajectory=None,
                 n_steps=int(hold_s / DT), grip_ratio=grip,
                 on_enter=enter, on_step=step, on_exit=on_exit,
                 monitor_force=monitor_force)


# ============================================================
# 阶段 1：抓取充电枪
# ============================================================
def build_phase1():
    phases = []
    # 1a: ee_link → gun_site_2（张开）
    phases.append(_move_phase("1a_to_gun_site_2", "运动到 gun_site_2",
                              Task.GUN_SITE_2, GRIP_OPEN, 2.0))
    # 1b: hold 1s @ gun_site_2
    phases.append(_hold_phase("1b_hold_gun_site_2", "停留 gun_site_2",
                              Task.GUN_SITE_2, GRIP_OPEN, 1.0))
    # 1c: → gun_site_1（张开）
    phases.append(_move_phase("1c_to_gun_site_1", "运动到 gun_site_1",
                              Task.GUN_SITE_1, GRIP_OPEN, 1.5))
    # 1d: hold 1s @ gun_site_1
    phases.append(_hold_phase("1d_hold_gun_site_1", "停留 gun_site_1",
                              Task.GUN_SITE_1, GRIP_OPEN, 1.0))

    # 1e: 闭合夹爪 hold + 抓取检测 + 耦合
    def exit_grasp(c):
        f = c.env.gripper.get_contact_force()
        if f >= GRASP_FORCE_MIN:
            # 解除插座 weld
            c.constraints.set_active(Task.EQ_SOCKET, False)
            # 耦合枪到末端（运动学跟随）
            c.coupler.attach()
            dt = _ctrl_dt(c)
            c.env.add_post_step_hook(
                lambda: (c.coupler.update(dt),
                         mujoco.mj_forward(c.env.model, c.env.data)))
            mujoco.mj_forward(c.env.model, c.env.data)
            # 记录位置偏移（ee_link 局部系，恒定）与旋转关系
            gun_p, gun_mat = c.env.site_pose(Task.GUN_SITE)
            ee_p = c.env.data.site_xpos[c.env.ee_site_id]
            ee_mat = c.env.data.site_xmat[c.env.ee_site_id].reshape(3, 3)
            # 局部系偏移 = R_ee^T @ (gun_world - ee_world)，不随姿态变化
            c.ee_to_gun_offset = ee_mat.T @ (gun_p - ee_p)
            c.grasp_rot = gun_mat @ ee_mat.T  # R_gun_ee，用于姿态对齐
            c.phase_msg = f"抓取成功 f={f:.2f}N 局部偏移={c.ee_to_gun_offset}"
        else:
            c.phase_done = True
            c.phase_msg = f"抓取失败 f={f:.2f}N < {GRASP_FORCE_MIN}N"

    phases.append(_hold_phase("1e_close_grasp", "闭合夹爪抓取",
                              Task.GUN_SITE_1, GRIP_CLOSE, 1.0,
                              on_exit=exit_grasp, monitor_force=True))
    return phases


# ============================================================
# 阶段 2：移动（gun_site → charing_site_2 → car_site_2）
# ============================================================
def build_phase2():
    phases = []
    # 抓取后所有运动用 offset_gun=True（ee 目标 = gun目标 - offset）
    phases.append(_move_phase("2a_to_charging_site_2", "gun_site→charing_site_2",
                              Task.CHARGING_SITE_2, GRIP_CLOSE, 2.0, offset_gun=True))
    phases.append(_hold_phase("2b_hold_charging_site_2", "停留 charing_site_2",
                              Task.CHARGING_SITE_2, GRIP_CLOSE, 1.0))
    # 2c: 大距离横移，经 via point 拆分为两小段，避免单段关节变化过大
    # via point 选在臂基座上方安全高度，两侧均可达且关节变化小
    VIA_POS = [0.0, -0.3, 1.15]
    phases.append(_move_pos_phase("2c1_via", "via point 过渡", VIA_POS,
                                  GRIP_CLOSE, 2.5, offset_gun=True))
    phases.append(_move_phase("2c2_to_car_site_2", "gun_site→car_site_2",
                              Task.CAR_SITE_2, GRIP_CLOSE, 2.5, offset_gun=True))
    phases.append(_hold_phase("2d_hold_car_site_2", "停留 car_site_2",
                              Task.CAR_SITE_2, GRIP_CLOSE, 1.0))
    return phases


# ============================================================
# 阶段 3：插枪（导纳控制 + 到底检测）
# ============================================================
def build_phase3():
    st = {"nominal": None, "blocked_n": 0, "prev_dist": 1e9, "at_target": False}

    def enter(c):
        mujoco.mj_forward(c.env.model, c.env.data)
        c.admittance.reset()
        # 名义 gun 目标从当前 gun_site 位置开始
        st["nominal"] = c.env.site_pose(Task.GUN_SITE)[0].copy()
        st["blocked_n"] = 0
        st["prev_dist"] = 1e9
        st["at_target"] = False
        c.forces.clear()

    def step(c, i):
        # 检测枪与车插座间的接触力
        f_contact = c.env.contact_force_between(Task.GUN_BODY, Task.CAR_SOCKET_BODY)
        f_mag = float(np.linalg.norm(f_contact))
        car1, car1_mat = c.env.site_pose(Task.CAR_SITE_1)
        car_done = c.env.site_pose(Task.CAR_SITE_DONE)[0]
        if f_mag < F_BLOCK:
            # 阻力可接受：力相关步长缩放推进（力越大步长越小，平滑柔顺）
            scale = max(0.1, 1.0 - f_mag / F_BLOCK)
            s = INSERT_STEP * scale
            # 先水平对齐 car_site_1（插座口中心），再沿插座轴向 car_site_done 推进
            horiz = car1 - st["nominal"]
            horiz[2] = 0
            if np.linalg.norm(horiz) > DONE_DIST:
                st["nominal"] = st["nominal"] + s * horiz / np.linalg.norm(horiz)
            else:
                d = car_done - st["nominal"]
                nd = np.linalg.norm(d)
                if nd > DONE_DIST:
                    st["nominal"] = st["nominal"] + s * d / nd
                else:
                    st["at_target"] = True
        else:
            # 受阻：暂停主推进，朝目标 xy 小范围横向调整寻找突破口
            err_xy = car1[:2] - st["nominal"][:2]
            ne = np.linalg.norm(err_xy)
            if ne > 1e-4:
                st["nominal"][:2] += SEARCH_STEP * err_xy / ne
        # 导纳修正：接触力产生顺从偏移（外力越大退让越多）
        c.admittance.step(f_contact, c.dt_ctrl)
        actual_gun = st["nominal"] + c.admittance.delta
        q_cur = c.env.data.qpos[c.env.arm_qposadr].copy()
        # 对 gun_site 求 IK，同时用 z 轴对齐约束枪体姿态（小步长微调，
        # 不约束绕 z 自转，避免大腕关节旋转）
        q_des = c.ik.solve(actual_gun, target_rot=car1_mat, q_init=q_cur,
                           site_id=c.gun_site_id, z_align_only=True)
        return q_des

    def done(c):
        # 到底：gun_col_6 碰 car_socket
        if c.env.geom_body_collides(Task.GUN_COL_6, Task.CAR_SOCKET_BODY):
            c.phase_msg = "插枪到底：gun_col_6 接触 car_socket"
            return True
        # 或 gun_site 到达 car_site_done
        gun_p = c.env.site_pose(Task.GUN_SITE)[0]
        done_p = c.env.site_pose(Task.CAR_SITE_DONE)[0]
        dist = np.linalg.norm(gun_p - done_p)
        if dist < DONE_DIST:
            c.phase_msg = "插枪到底：到达 car_site_done"
            return True
        # 停滞检测：枪体接触 car_socket 且距离不再减小
        if c.env.body_collides_with(Task.GUN_BODY, Task.CAR_SOCKET_BODY):
            if dist < st["prev_dist"] - 1e-4:
                st["blocked_n"] = 0
            else:
                st["blocked_n"] += 1
            st["prev_dist"] = min(st["prev_dist"], dist)
            if st["blocked_n"] > 30:  # 约 1.5s 无进展
                c.phase_msg = "插枪停滞：接触 car_socket 且无进展"
                return True
        return False

    return [Phase(name="phase3_insert", desc="导纳插枪",
                  trajectory=None, n_steps=INSERT_MAX_STEPS,
                  grip_ratio=GRIP_CLOSE, on_enter=enter, on_step=step,
                  done_condition=done, monitor_force=True, use_admittance=True)]


# ============================================================
# 阶段 4：拔枪（gun_site → car_site_2 → charing_site_2）
# ============================================================
def build_phase4():
    phases = []
    phases.append(_move_phase("4a_to_car_site_2", "拔枪→car_site_2",
                              Task.CAR_SITE_2, GRIP_CLOSE, 1.5, offset_gun=True))
    phases.append(_hold_phase("4b_hold_car_site_2", "停留 car_site_2",
                              Task.CAR_SITE_2, GRIP_CLOSE, 1.0))
    # 4c: 经 via point 拆分大距离横移，避免单段关节变化过大
    VIA_POS = [0.0, -0.3, 1.15]
    phases.append(_move_pos_phase("4c1_via", "via point 过渡", VIA_POS,
                                  GRIP_CLOSE, 2.5, offset_gun=True))
    phases.append(_move_phase("4c2_to_charging_site_2", "→charing_site_2",
                              Task.CHARGING_SITE_2, GRIP_CLOSE, 2.5, offset_gun=True))
    return phases


# ============================================================
# 阶段 5：枪体归位（导纳 → charing_site_1 → 下移 → 激活 weld）
# ============================================================
def build_phase5():
    # 5a: 导纳控制 gun_site → charing_site_1
    st = {"nominal": None}

    def enter_a(c):
        mujoco.mj_forward(c.env.model, c.env.data)
        c.admittance.reset()
        st["nominal"] = c.env.site_pose(Task.GUN_SITE)[0].copy()
        gun0 = st["nominal"].copy()
        # 4c 终点 gun_col_6 可能已穿入充电插座（gun_site 在 charing_site_2，
        # gun_col_6 在其下方 0.1m，沿插座轴深入插座内部）。若已重叠，
        # 物理后退枪体 2cm 脱离接触，避免第一步解算 penetration 产生峰值力
        collided = c.env.geom_body_collides(Task.GUN_COL_6, Task.SOCKET_BODY)
        ee_now = c.env.data.site_xpos[c.env.ee_site_id].copy()
        ee_mat_now = c.env.data.site_xmat[c.env.ee_site_id].reshape(3, 3)
        off_local = ee_mat_now.T @ (gun0 - ee_now)
        print(f"  [5a enter] gun_site={np.round(gun0,4)} ee={np.round(ee_now,4)} "
              f"局部偏移={np.round(off_local,4)} 记录偏移={np.round(c.ee_to_gun_offset,4)} "
              f"碰插座={collided}")
        if collided:
            _, cs1_mat = c.env.site_pose(Task.CHARGING_SITE_1)
            # gun_col_6 在 gun_site 下方约 0.1m，后退 0.05m 沿插座 z 轴脱离接触
            # 用纯位置 IK（不用 z_align_only）避免 180° 翻转改变臂构型
            back_pos = st["nominal"] + 0.05 * cs1_mat[:, 2]
            q_cur = c.env.data.qpos[c.env.arm_qposadr].copy()
            q_back = c.ik.solve(back_pos, target_rot=None, q_init=q_cur,
                                site_id=c.gun_site_id)
            c.env.data.qpos[c.env.arm_qposadr] = q_back
            # 必须先 mj_forward 更新 xpos，coupler.update 才能读到正确的锚点位姿
            mujoco.mj_forward(c.env.model, c.env.data)
            c.coupler.update(c.dt_ctrl)
            mujoco.mj_forward(c.env.model, c.env.data)
            gun_after = c.env.site_pose(Task.GUN_SITE)[0].copy()
            gun_mat_after = c.env.site_pose(Task.GUN_SITE)[1].copy()
            still_collided = c.env.geom_body_collides(Task.GUN_COL_6, Task.SOCKET_BODY)
            any_collided = c.env.body_collides_with(Task.GUN_BODY, Task.SOCKET_BODY)
            f_ret = float(np.linalg.norm(
                c.env.contact_force_between(Task.GUN_BODY, Task.SOCKET_BODY)))
            st["nominal"] = gun_after
            print(f"  [5a 后退后] gun_site={np.round(gun_after,4)} "
                  f"gun_z={np.round(gun_mat_after[:,2],4)} "
                  f"gun_col_6碰={still_collided} 任意碰={any_collided} 力={f_ret:.1f}N")

    def step_a(c, i):
        # 检测枪与充电插座间的接触力
        f_contact = c.env.contact_force_between(Task.GUN_BODY, Task.SOCKET_BODY)
        f_mag = float(np.linalg.norm(f_contact))
        cs1, cs1_mat = c.env.site_pose(Task.CHARGING_SITE_1)
        if f_mag < F_BLOCK_SLOW:
            # 阻力可接受：力相关步长缩放慢速推进
            scale = max(0.1, 1.0 - f_mag / F_BLOCK_SLOW)
            s = INSERT_STEP_SLOW * scale
            d = cs1 - st["nominal"]
            nd = np.linalg.norm(d)
            if nd > DONE_DIST:
                st["nominal"] = st["nominal"] + s * d / nd
        else:
            # 受阻：沿接触力方向后退（力推枪离开插座），避免力持续累积
            # gun_col_6 半径 0.05 > 插座开口，无法深入，后退保力安全
            if f_mag > 1e-3:
                st["nominal"] += 0.002 * f_contact / f_mag
        c.admittance.step(f_contact, c.dt_ctrl)
        actual = st["nominal"] + c.admittance.delta
        # 限制每步 IK 目标变化量，防止 admittance delta 累积饱和后目标跳变
        # 导致臂大幅运动撞插座（峰值力 2800N+ → 降到安全范围）
        prev_actual = st.get("prev_actual", actual.copy())
        MAX_TARGET_STEP = 0.001  # 每步最多 1mm
        diff = actual - prev_actual
        dnorm = np.linalg.norm(diff)
        if dnorm > MAX_TARGET_STEP:
            actual = prev_actual + MAX_TARGET_STEP * diff / dnorm
        st["prev_actual"] = actual.copy()
        q_cur = c.env.data.qpos[c.env.arm_qposadr].copy()
        # 纯位置 IK：姿态已由 4c2 对齐，导纳段每步仅移动 0.6mm，
        # 用位置 IK 避免两阶段姿态求解导致臂构型跳变
        q_des = c.ik.solve(actual, target_rot=None, q_init=q_cur,
                           site_id=c.gun_site_id)
        if i < 3 or f_mag > 10:
            dq = np.max(np.abs(q_des - q_cur))
            gun_now = c.env.site_pose(Task.GUN_SITE)[0]
            print(f"  [5a step {i}] f={f_mag:.2f} nominal={np.round(st['nominal'],4)} "
                  f"actual={np.round(actual,4)} gun={np.round(gun_now,4)} "
                  f"max|Δq|={dq:.4f}rad")
        return q_des

    def done_a(c):
        # 力安全限位：超过 80N 立即停止（gun_col_6 比插座开口宽，深入会力暴涨）
        f_mag = float(np.linalg.norm(
            c.env.contact_force_between(Task.GUN_BODY, Task.SOCKET_BODY)))
        if f_mag > 80.0:
            c.phase_msg = f"枪体归位：力安全限位 ({f_mag:.1f}N)"
            return True
        # 主条件：gun_site 到达 charing_site_1 附近（5cm 内即可，weld 会固定枪位）
        gun_p = c.env.site_pose(Task.GUN_SITE)[0]
        cs1 = c.env.site_pose(Task.CHARGING_SITE_1)[0]
        dist = np.linalg.norm(gun_p - cs1)
        if dist < 0.05:
            c.phase_msg = f"枪体归位：接近 charing_site_1 ({dist:.4f}m)"
            return True
        # 停滞检测：枪接触充电插座且无进展
        in_contact = (c.env.geom_body_collides(Task.GUN_COL_6, Task.SOCKET_BODY) or
                      c.env.body_collides_with(Task.GUN_BODY, Task.SOCKET_BODY))
        if in_contact:
            if dist < st.get("prev_dist_a", 1e9) - 1e-4:
                st["blocked_a"] = 0
            else:
                st["blocked_a"] = st.get("blocked_a", 0) + 1
            st["prev_dist_a"] = min(st.get("prev_dist_a", 1e9), dist)
            if st.get("blocked_a", 0) > 30:
                c.phase_msg = f"枪体归位：停滞（接触插座，距 cs1={dist:.4f}m）"
                return True
        return False

    # 5b: 沿 charing_site_1 z 负方向 0.01m（gun_site 下移）
    stb = {"traj": None}

    def enter_b(c):
        mujoco.mj_forward(c.env.model, c.env.data)
        cs1_pos, cs1_mat = c.env.site_pose(Task.CHARGING_SITE_1)
        gun_p = c.env.site_pose(Task.GUN_SITE)[0]
        in_contact = c.env.geom_body_collides(Task.GUN_COL_6, Task.SOCKET_BODY)
        print(f"  [5a结束] gun_site={np.round(gun_p,4)} cs1={np.round(cs1_pos,4)} "
              f"距离={np.linalg.norm(gun_p - cs1_pos):.4f}m 碰插座={in_contact}")
        if in_contact:
            # gun_col_6 已碰插座（开口比枪头窄），下移会增力；保持原位，weld 固定
            stb["traj"] = _hold_traj(
                c.env.data.qpos[c.env.arm_qposadr].copy(), int(1.0 / DT), c.dt_ctrl)
        else:
            # charing_site_1 局部 -z 方向（世界系）
            neg_z = -cs1_mat[:, 2]
            gun_target = cs1_pos + neg_z * 0.01  # gun_site 目标
            q0 = c.env.data.qpos[c.env.arm_qposadr].copy()
            traj, _ = _plan(c, gun_target, None, 1.0, q_init=q0,
                            site_id=c.gun_site_id)
            stb["traj"] = traj

    def step_b(c, i):
        return stb["traj"].at(i)

    # 5b 结束后激活插座 weld + 解除耦合
    def exit_b(c):
        c.coupler.detach()
        c.env.clear_post_step_hooks()
        c.constraints.set_active(Task.EQ_SOCKET, True)
        mujoco.mj_forward(c.env.model, c.env.data)
        c.phase_msg = "枪体归位，激活插座 weld"

    return [
        Phase(name="phase5a_admittance_home", desc="导纳归位到 charing_site_1",
              trajectory=None, n_steps=INSERT_MAX_STEPS, grip_ratio=GRIP_CLOSE,
              on_enter=enter_a, on_step=step_a, done_condition=done_a,
              monitor_force=True, use_admittance=True),
        Phase(name="phase5b_insert_down", desc="沿 z 负方向下移 0.01m",
              trajectory=None, n_steps=int(1.0 / DT), grip_ratio=GRIP_CLOSE,
              on_enter=enter_b, on_step=step_b, on_exit=exit_b),
    ]


# ============================================================
# 阶段 6：机械臂复位（松开夹爪 → 回 home）
# ============================================================
def build_phase6(home_qpos):
    st = {"traj": None}

    # 6a: 松开夹爪 hold
    def enter_open(c):
        pass

    def step_open(c, i):
        return c.env.data.qpos[c.env.arm_qposadr].copy()

    # 6b: 回 home
    def enter_home(c):
        q0 = c.env.data.qpos[c.env.arm_qposadr].copy()
        wp = min_jerk(q0, home_qpos, 2.0, c.dt_ctrl)
        st["traj"] = Trajectory(wp, c.dt_ctrl)

    def step_home(c, i):
        return st["traj"].at(i)

    return [
        Phase(name="phase6a_open_gripper", desc="松开夹爪",
              trajectory=None, n_steps=HOLD_1S, grip_ratio=GRIP_OPEN,
              on_enter=enter_open, on_step=step_open),
        Phase(name="phase6b_home", desc="机械臂复位",
              trajectory=None, n_steps=int(2.0 / DT), grip_ratio=GRIP_OPEN,
              on_enter=enter_home, on_step=step_home),
    ]


# ---- 汇总 ----
def build_all_phases(ctx: PhaseContext, home_qpos) -> list:
    """构建全部 6 阶段的所有子段 Phase。"""
    return (build_phase1() + build_phase2() + build_phase3()
            + build_phase4() + build_phase5() + build_phase6(home_qpos))
