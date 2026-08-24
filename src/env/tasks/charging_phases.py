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
GRASP_FORCE_MIN = 0.5      # 抓取成功最小夹爪力 (N)；枪柄 geom 加宽后
                           # 稳态夹持力 ~0.9N，阈值 1.0 会误判失败（枪仅
                           # ~0.1kg，0.9N 夹持 + 运动学耦合足够可靠）
F_BLOCK = 5.0              # 插枪阻力阈值 (N)，超过则暂停主推进
INSERT_STEP = 0.002        # 插枪名义推进步长 (m)
INSERT_MAX_STEPS = 400     # 插枪段最大步数
DONE_DIST = 0.006          # 到底距离阈值 (m)
ALIGN_TOL = 0.0015         # 对心阈值：实际横向偏差低于此值才推进 (m)
                           # （弹片口与枪头间隙仅 2.5mm，必须先对心）
ALIGN_STEP = 0.001         # 对心闭环步长 (m/步)
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
    记录 ctx.last_ik_q / last_ik_target 供 run 脚本做轨迹终点一致性校验。
    """
    q_cur = (q_init if q_init is not None
             else ctx.env.data.qpos[ctx.env.arm_qposadr].copy())
    q_goal = ctx.ik.solve(target_pos, target_rot=target_rot, q_init=q_cur,
                          site_id=site_id, z_align_only=z_align_only)
    ctx.last_ik_q = q_goal.copy()
    ctx.last_ik_target = np.asarray(target_pos, float).copy()
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
                target_rot_site=None, full_rot=False, back_along_z=0.0,
                on_enter_extra=None, on_exit=None, monitor_force=False):
    """生成运动到指定 site 的 Phase。轨迹在 on_enter 规划。

    target_rot_site: 若指定，终点 IK 同时约束姿态。
    full_rot=False: 仅 z 轴对齐（绕 z 自转自由），适合插座插拔对位；
    full_rot=True: 全姿态匹配（x/y/z 三轴都对齐），抓取枪柄时必须用
    ——手指相对柄的取向由绕 z 自转决定，不锁定会抓偏。
    back_along_z: >0 时目标点 = site 位置沿 site z 轴后退该距离（预接近点，
    site z 轴指向枪体/插座内侧，后退即远离），用于先在安全距离转正姿态。
    轨迹末尾追加 settle 保持段（SETTLE_S 秒停在 q_goal）：若 q_goal 仅在
    最后一步被指令，position controller 无收敛时间，滞后直接计入误差。
    phase.length 取轨迹全长（on_enter 中回填 Phase.trajectory）。
    """
    st = {"traj": None, "phase": None}
    SETTLE_S = 1.0

    def enter(c):
        mujoco.mj_forward(c.env.model, c.env.data)
        q0 = c.env.data.qpos[c.env.arm_qposadr].copy()
        pos, site_m = c.env.site_pose(site_name)
        if back_along_z > 0:
            # 预接近点：沿 site z 轴后退（z 轴指向枪体/插座内侧）
            pos = pos - back_along_z * site_m[:, 2]
        # offset_gun=True 时直接对 gun_site 求 IK（利用枪体延伸臂展），
        # target_rot 直接用目标 site 旋转矩阵（IK 匹配 gun_site 姿态）
        sid = c.gun_site_id if offset_gun else None
        target_rot = None
        z_align = False
        if target_rot_site is not None:
            _, site_mat = c.env.site_pose(target_rot_site)
            target_rot = site_mat if offset_gun else \
                _gun_rot_to_ee(c, site_mat)
            # full_rot=True 全姿态匹配；False 仅 z 轴对齐（绕 z 自转自由）
            z_align = not full_rot
        target = pos  # gun_site 目标 = site 世界位置（back_along_z>0 时含偏移）
        c.last_plan_name = name  # 供 run 脚本用规划目标（含偏移）做对比
        traj, q_goal = _plan(c, target, target_rot, T, q_init=q0,
                             site_id=sid, z_align_only=z_align)
        # 追加 settle 段：重复 q_goal 等待控制器收敛
        n_settle = max(1, int(SETTLE_S / c.dt_ctrl))
        wp = np.vstack([traj.wp, np.tile(q_goal, (n_settle, 1))])
        st["traj"] = Trajectory(wp, c.dt_ctrl)
        if st["phase"] is not None:
            st["phase"].trajectory = st["traj"]  # length = 轨迹全长
        # 诊断：关节角度差异（检测构型翻转）
        dq_max = np.max(np.abs(q_goal - q0))
        print(f"  [JNT] {name}: q0={np.round(q0,3)} q_goal={np.round(q_goal,3)} "
              f"max|Δq|={dq_max:.3f}rad")
        if on_enter_extra:
            on_enter_extra(c)

    def step(c, i):
        if st["traj"] is None:
            return c.env.data.qpos[c.env.arm_qposadr].copy()
        return st["traj"].at(i)

    ph = Phase(name=name, desc=desc, trajectory=None, n_steps=int(T / DT),
               grip_ratio=grip, on_enter=enter, on_step=step,
               on_exit=on_exit, monitor_force=monitor_force)
    st["phase"] = ph
    return ph


def _move_pos_phase(name, desc, target_pos, grip, T, offset_gun=True):
    """运动到指定世界坐标（非 site）的 Phase。用于 via point 过渡。

    target_pos 可为 (3,) 数组，或无参 callable -> (3,)（进入阶段时求值，
    适合依赖运行时 site 位姿的 via 点，如两端点中点）。
    与 _move_phase 类似但接受位置而非 site 名，纯位置 IK（不约束姿态）。
    同样追加 settle 保持段，phase.length 取轨迹全长。
    """
    st = {"traj": None, "phase": None}
    SETTLE_S = 1.0

    def enter(c):
        target = target_pos(c) if callable(target_pos) \
            else np.asarray(target_pos, float)
        mujoco.mj_forward(c.env.model, c.env.data)
        q0 = c.env.data.qpos[c.env.arm_qposadr].copy()
        sid = c.gun_site_id if offset_gun else None
        traj, q_goal = _plan(c, target, None, T, q_init=q0, site_id=sid)
        n_settle = max(1, int(SETTLE_S / c.dt_ctrl))
        wp = np.vstack([traj.wp, np.tile(q_goal, (n_settle, 1))])
        st["traj"] = Trajectory(wp, c.dt_ctrl)
        if st["phase"] is not None:
            st["phase"].trajectory = st["traj"]
        dq_max = np.max(np.abs(q_goal - q0))
        print(f"  [JNT] {name}: max|Δq|={dq_max:.3f}rad via={np.round(target,3)}")

    def step(c, i):
        if st["traj"] is None:
            return c.env.data.qpos[c.env.arm_qposadr].copy()
        return st["traj"].at(i)

    ph = Phase(name=name, desc=desc, trajectory=None, n_steps=int(T / DT),
               grip_ratio=grip, on_enter=enter, on_step=step)
    st["phase"] = ph
    return ph


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
    # 1a-0: 预接近点（gun_site_2 沿枪轴外退 15cm），先在此处完成全姿态
    # 对齐。远离枪体转动腕关节安全，避免接近途中姿态未转正手指扫到枪
    phases.append(_move_phase("1a0_pre", "预接近（枪轴外 15cm 转正姿态）",
                              Task.GUN_SITE_2, GRIP_OPEN, 3.0,
                              target_rot_site=Task.GUN_SITE_2, full_rot=True,
                              back_along_z=0.15))
    # 1a: 预接近点 → gun_site_2，短距离沿枪轴推进，姿态已对齐且保持不变
    phases.append(_move_phase("1a_to_gun_site_2", "运动到 gun_site_2",
                              Task.GUN_SITE_2, GRIP_OPEN, 1.5,
                              target_rot_site=Task.GUN_SITE_2, full_rot=True))
    # 1b: hold 1s @ gun_site_2
    phases.append(_hold_phase("1b_hold_gun_site_2", "停留 gun_site_2",
                              Task.GUN_SITE_2, GRIP_OPEN, 1.0))
    # 1c: → gun_site_1（张开），全姿态对齐（抓取点，取向必须精确）
    phases.append(_move_phase("1c_to_gun_site_1", "运动到 gun_site_1",
                              Task.GUN_SITE_1, GRIP_OPEN, 2.5,
                              target_rot_site=Task.GUN_SITE_1, full_rot=True))
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
    # via point 动态取 charing_site_2 与 car_site_2 世界坐标中点（进入阶段时
    # 求值）——固定常量曾取 [0,-0.3,1.15] 落在臂体上导致 IK 不收敛
    def via_mid(c):
        p1 = c.env.site_pose(Task.CHARGING_SITE_2)[0]
        p2 = c.env.site_pose(Task.CAR_SITE_2)[0]
        return (p1 + p2) / 2.0
    phases.append(_move_pos_phase("2c1_via", "via point 过渡", via_mid,
                                  GRIP_CLOSE, 2.5, offset_gun=True))
    # 2c2 终点加 z 轴对齐：到达 car_site_2（插座口外预接近点）时枪姿
    # 态已转正。若纯位置到达，姿态任意（可能反向），phase3 的 z_align
    # 第一步要原地翻转 π，甩枪撞击插座（曾见瞬时 3 万 N 冲击）
    phases.append(_move_phase("2c2_to_car_site_2", "gun_site→car_site_2",
                              Task.CAR_SITE_2, GRIP_CLOSE, 2.5, offset_gun=True,
                              target_rot_site=Task.CAR_SITE_2))
    phases.append(_hold_phase("2d_hold_car_site_2", "停留 car_site_2",
                              Task.CAR_SITE_2, GRIP_CLOSE, 1.0))
    return phases


# ============================================================
# 阶段 3：插枪（导纳控制 + 到底检测）
# ============================================================
def build_phase3():
    st = {"nominal": None, "blocked_n": 0, "prev_dist": 1e9, "at_target": False,
          "prev_q_des": None, "prev_actual": None}

    def enter(c):
        mujoco.mj_forward(c.env.model, c.env.data)
        c.admittance.reset()
        # 名义 gun 目标从当前 gun_site 位置开始
        st["nominal"] = c.env.site_pose(Task.GUN_SITE)[0].copy()
        # 诊断：进入插枪段时枪轴与插座轴的夹角（间隙仅 ~2.5mm，
        # 夹角 >0.5° 时 0.4m 枪长引起 >3mm 横偏，会卡口刮蹭）
        _, gun_m = c.env.site_pose(Task.GUN_SITE)
        _, car_m = c.env.site_pose(Task.CAR_SITE_1)
        cosz = float(np.clip(gun_m[:, 2] @ car_m[:, 2], -1.0, 1.0))
        print(f"  [3 enter] 枪轴vs插座轴夹角={np.degrees(np.arccos(abs(cosz))):.2f}deg "
              f"枪头位置偏差={np.round(st['nominal'] - c.env.site_pose(Task.CAR_SITE_1)[0], 4)}")
        st["blocked_n"] = 0
        st["prev_dist"] = 1e9
        st["at_target"] = False
        st["prev_q_des"] = None
        st["prev_actual"] = None
        c.forces.clear()

    def step(c, i):
        # 检测枪与车插座间的接触力
        f_contact = c.env.contact_force_between(Task.GUN_BODY, Task.CAR_SOCKET_BODY)
        f_mag = float(np.linalg.norm(f_contact))
        car1, car1_mat = c.env.site_pose(Task.CAR_SITE_1)
        car_done = c.env.site_pose(Task.CAR_SITE_DONE)[0]
        ax = car1_mat[:, 2]  # 插座轴（世界系）
        # 实际枪头相对插座轴的横向偏差（闭环反馈量——只对准 nominal
        # 不够：跟踪误差使实际偏差 4~6mm，而弹片口间隙仅 2.5mm，
        # 超限即卡口刮蹭、接触力推臂越偏越远死循环）
        gun_act, _ = c.env.site_pose(Task.GUN_SITE)
        r_ = gun_act - car1
        ax_comp = float(r_ @ ax)
        lat = r_ - ax_comp * ax
        lat_norm = np.linalg.norm(lat)
        if lat_norm > ALIGN_TOL:
            # 对心优先（暂停轴向推进）：nominal 朝消除实际横向偏差移动
            st["nominal"] -= ALIGN_STEP * lat / lat_norm
            if f_mag > F_BLOCK and f_mag > 1e-3:
                st["nominal"] += 0.001 * f_contact / f_mag
        elif f_mag < F_BLOCK:
            # 横向已对准且阻力可接受：力相关步长缩放沿轴推进
            scale = max(0.1, 1.0 - f_mag / F_BLOCK)
            s = INSERT_STEP * scale
            d = car_done - st["nominal"]
            nd = np.linalg.norm(d)
            if nd > DONE_DIST:
                st["nominal"] = st["nominal"] + s * d / nd
            else:
                st["at_target"] = True
        else:
            # 已对准但受阻：沿接触力方向回退卸力
            if f_mag > 1e-3:
                st["nominal"] += 0.001 * f_contact / f_mag
        # 导纳修正：接触力产生顺从偏移（外力越大退让越多）
        c.admittance.step(f_contact, c.dt_ctrl)
        # 导纳偏移投影到插座轴向：弹片斜置使接触力带切向分量，横向
        # 顺从会把枪推离轴心（曾漂移 1.2cm）导致 col_1 卡死在弹片口
        # 越卡力越大、偏移越大死循环。深插段只保留沿轴退让
        d_ = c.admittance.delta
        c.admittance.delta[:] = ax * float(d_ @ ax)
        v_ = c.admittance.delta_dot
        c.admittance.delta_dot[:] = ax * float(v_ @ ax)
        actual_gun = st["nominal"] + c.admittance.delta
        # 目标变化限幅：碰撞瞬间 nominal 搜索/导纳 delta 若有残余突变，
        # 截断单步目标变化量，防止 IK 输入跳变传导为 q_des 跳变
        prev_actual = st["prev_actual"]
        if prev_actual is not None:
            MAX_TARGET_STEP = 0.003  # 容纳 nominal 2mm + 导纳 0.5mm + 裕量
            diff = actual_gun - prev_actual
            dnorm = np.linalg.norm(diff)
            if dnorm > MAX_TARGET_STEP:
                actual_gun = prev_actual + MAX_TARGET_STEP * diff / dnorm
        st["prev_actual"] = actual_gun.copy()
        # q_init 链式使用上一步 q_des（首步用当前实际关节角）：
        # 若用滞后的 qpos 实际值，IK 每步都要"追赶"跟踪滞后，输出偏大；
        # 链式传递让连续两步的解天然连续
        q_cur = c.env.data.qpos[c.env.arm_qposadr].copy()
        _prev_q = st.get("prev_q_des")
        q_init = _prev_q if _prev_q is not None else q_cur
        # 对 gun_site 求 IK，同时用 z 轴对齐约束枪体姿态（小步长微调，
        # 不约束绕 z 自转，避免大腕关节旋转）。retry=False：闭环实时段
        # 禁用多起点重试；max_travel=0.1：姿态欠收敛时 IK 零空间漂移
        # 曾达 0.76rad/步，硬性截断保证单步 q_des 变化有界
        q_des = c.ik.solve(actual_gun, target_rot=car1_mat, q_init=q_init,
                           site_id=c.gun_site_id, z_align_only=True,
                           retry=False, max_travel=0.1)
        st["prev_q_des"] = q_des.copy()
        # 低频诊断：接触力、名义目标、导纳偏移、单步关节变化量
        if i % 20 == 0 or f_mag > 100:
            dq = float(np.max(np.abs(q_des - q_cur)))
            # 列出枪-插座当前接触的 geom 对（定位是哪个部件在顶）
            pairs = []
            if f_mag > 1.0:
                gnames = [mujoco.mj_id2name(c.env.model, mujoco.mjtObj.mjOBJ_GEOM,
                                            g) for g in range(c.env.model.ngeom)]
                b2 = mujoco.mj_name2id(c.env.model, mujoco.mjtObj.mjOBJ_BODY,
                                       Task.CAR_SOCKET_BODY)
                g2s = set(range(c.env.model.body_geomadr[b2],
                                c.env.model.body_geomadr[b2] +
                                c.env.model.body_geomnum[b2]))
                for ci in range(c.env.data.ncon):
                    ct = c.env.data.contact[ci]
                    for a, b in ((ct.geom1, ct.geom2), (ct.geom2, ct.geom1)):
                        if b in g2s and gnames[a] and gnames[a].startswith("gun"):
                            pairs.append(f"{gnames[a]}|{gnames[b]}")
                            break
            print(f"  [3 step {i}] f={f_mag:.1f} "
                  f"nominal={np.round(st['nominal'],4)} "
                  f"delta={np.round(c.admittance.delta,4)} "
                  f"max|Δq|={dq:.3f}rad pairs={sorted(set(pairs))[:4]}")
            # 实际枪头的横向偏移（相对插座轴）：间隙仅 2.5mm，超出即卡口
            gun_now, gun_m_now = c.env.site_pose(Task.GUN_SITE)
            r_ = gun_now - car1
            ax_comp = float(r_ @ ax)
            lat_ = r_ - ax_comp * ax
            gz = gun_m_now[:, 2]
            ang_ = float(np.degrees(np.arccos(abs(float(gz @ ax)))))
            print(f"           实际横向偏差={np.linalg.norm(lat_)*1000:.1f}mm "
                  f"轴向深度={ax_comp*1000:+.1f}mm 实际夹角={ang_:.2f}deg")
        return q_des

    def done(c):
        # 插入成功：gun_pan（枪头圆盘）接触 car_socket。
        # 流程：gun_site 先对齐 car_site_1（插座口），再沿轴向推进到
        # car_site_done；枪头深入后 gun_pan 碰到插座即插入到底
        if c.env.body_collides_with(Task.GUN_PAN_BODY, Task.CAR_SOCKET_BODY):
            c.phase_msg = "插枪到底：gun_pan 接触 car_socket"
            return True
        # 兜底：gun_site 到达 car_site_done
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
    # 4c: 经 via point 拆分大距离横移（同 2c，动态取两端点中点）
    def via_mid(c):
        p1 = c.env.site_pose(Task.CAR_SITE_2)[0]
        p2 = c.env.site_pose(Task.CHARGING_SITE_2)[0]
        return (p1 + p2) / 2.0
    phases.append(_move_pos_phase("4c1_via", "via point 过渡", via_mid,
                                  GRIP_CLOSE, 2.5, offset_gun=True))
    # 4c2 终点加 z 轴对齐（同 2c2）：纯位置 IK 不约束姿态，曾因大构型
    # 翻转（travel 6.4rad）导致枪 z 轴反向到达——gun_pan 粗盘朝插座内，
    # 在口外 71mm 时已深入弹片区卡死（90N），5a/5b 全部失效
    phases.append(_move_phase("4c2_to_charging_site_2", "→charing_site_2",
                              Task.CHARGING_SITE_2, GRIP_CLOSE, 2.5, offset_gun=True,
                              target_rot_site=Task.CHARGING_SITE_2))
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
            # 沿插座 -z（退出方向）后退 0.05m 脱离接触。
            # 注意方向：-ax 才是远离插座；曾误用 +ax 把枪顶得更深
            back_pos = st["nominal"] - 0.05 * cs1_mat[:, 2]
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
        ax_ = cs1_mat[:, 2]  # 插座轴，+z 为插入方向
        # 对心优先（同 phase3）：实际横向偏差超阈值时先横向修正。
        # 直接朝 cs1 直线推进会带着横向偏差卡弹片（曾偏 10.8mm 顶死
        # 91N，残留穿透让 5b 首步即误触发到位检测）
        gun_act, _ = c.env.site_pose(Task.GUN_SITE)
        r_ = gun_act - cs1
        lat = r_ - float(r_ @ ax_) * ax_
        lat_norm = np.linalg.norm(lat)
        if lat_norm > ALIGN_TOL:
            st["nominal"] -= ALIGN_STEP * lat / lat_norm
            if f_mag > F_BLOCK_SLOW and f_mag > 1e-3:
                st["nominal"] += 0.002 * f_contact / f_mag
        elif f_mag < F_BLOCK_SLOW:
            # 对心完成：力相关步长缩放慢速推进（朝 cs1）
            scale = max(0.1, 1.0 - f_mag / F_BLOCK_SLOW)
            s = INSERT_STEP_SLOW * scale
            d = cs1 - st["nominal"]
            nd = np.linalg.norm(d)
            if nd > DONE_DIST:
                st["nominal"] = st["nominal"] + s * d / nd
        else:
            # 已对准但受阻：沿接触力方向后退（力推枪离开插座），避免力持续累积
            if f_mag > 1e-3:
                st["nominal"] += 0.002 * f_contact / f_mag
        c.admittance.step(f_contact, c.dt_ctrl)
        # 导纳偏移投影到插座轴向（同 phase3）：弹片切向力会把枪横向
        # 推离轴心导致卡口，只保留沿轴退让分量
        dlt = c.admittance.delta
        c.admittance.delta[:] = ax_ * float(dlt @ ax_)
        vlt = c.admittance.delta_dot
        c.admittance.delta_dot[:] = ax_ * float(vlt @ ax_)
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
        # q_init 链式用上一步 q_des（同 phase3），避免跟踪滞后放大 IK 输出
        q_cur = c.env.data.qpos[c.env.arm_qposadr].copy()
        _prev_q = st.get("prev_q_des")
        q_init = _prev_q if _prev_q is not None else q_cur
        # 纯位置 IK：姿态已由 4c2 对齐，导纳段每步仅移动 0.6mm，
        # 用位置 IK 避免两阶段姿态求解导致臂构型跳变。
        # retry=False：闭环实时段禁用多起点重试，防 q_des 单步跳变；
        # max_travel：硬性限制单步解相对初值的行程上限
        # z_align 姿态约束（同 phase3）：纯位置 IK 不锁姿态，长行程推进
        # 中枪轴漂移（实测歪 11.6°，枪头横向偏 11mm 靠边缘蹭碰检测盘）。
        # 锁 z 轴平行保证轴向插入，枪头端面平顶到位盘
        q_des = c.ik.solve(actual, target_rot=cs1_mat, q_init=q_init,
                           site_id=c.gun_site_id, z_align_only=True,
                           retry=False, max_travel=0.1)
        st["prev_q_des"] = q_des.copy()
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

    # 5b: 沿 charing_site_1 z 正方向插入 0.1m，枪体接触 charing_pan 即到位
    stb = {"nominal": None, "prev_q_des": None, "prev_actual": None}

    def enter_b(c):
        mujoco.mj_forward(c.env.model, c.env.data)
        c.admittance.reset()
        stb["nominal"] = c.env.site_pose(Task.GUN_SITE)[0].copy()
        stb["prev_q_des"] = None
        stb["prev_actual"] = None
        gun_p = c.env.site_pose(Task.GUN_SITE)[0]
        cs1_pos, cs1_mat = c.env.site_pose(Task.CHARGING_SITE_1)
        depth0 = float((gun_p - cs1_pos) @ cs1_mat[:, 2])
        print(f"  [5a结束] gun_site={np.round(gun_p,4)} cs1={np.round(cs1_pos,4)} "
              f"初始轴向深度={depth0*1000:+.1f}mm 目标=沿+z插入100mm")

    def step_b(c, i):
        cs1_pos, cs1_mat = c.env.site_pose(Task.CHARGING_SITE_1)
        ax_ = cs1_mat[:, 2]  # 插座轴，+z 为插入方向
        target = cs1_pos + 0.1 * ax_  # gun_site 目标：沿 z 正方向 0.1m
        f_contact = c.env.contact_force_between(Task.GUN_BODY, Task.SOCKET_BODY)
        f_mag = float(np.linalg.norm(f_contact))
        # 对心优先（同 phase3）：弹片口间隙仅 ~2.5mm，实际横向偏差
        # 超阈值时先横向修正，暂停轴向推进
        gun_act, _ = c.env.site_pose(Task.GUN_SITE)
        r_ = gun_act - cs1_pos
        lat = r_ - float(r_ @ ax_) * ax_
        lat_norm = np.linalg.norm(lat)
        if lat_norm > ALIGN_TOL:
            stb["nominal"] -= ALIGN_STEP * lat / lat_norm
        elif float((target - stb["nominal"]) @ ax_) > INSERT_STEP_SLOW:
            # 对心完成：沿 +z 慢速推进（不超过目标深度）
            stb["nominal"] += INSERT_STEP_SLOW * ax_
        # 受阻回退卸力
        if f_mag > F_BLOCK_SLOW and f_mag > 1e-3:
            stb["nominal"] += 0.001 * f_contact / f_mag
        actual = stb["nominal"] + c.admittance.delta
        # 单步目标限幅，防 IK 输入跳变传导为 q_des 跳变
        prev_actual = stb["prev_actual"]
        if prev_actual is not None:
            diff = actual - prev_actual
            dnorm = np.linalg.norm(diff)
            if dnorm > 0.003:
                actual = prev_actual + 0.003 * diff / dnorm
        stb["prev_actual"] = actual.copy()
        q_cur = c.env.data.qpos[c.env.arm_qposadr].copy()
        _prev_q = stb.get("prev_q_des")
        q_init = _prev_q if _prev_q is not None else q_cur
        # z_align 姿态约束（同 5a/phase3）：锁枪轴与插座轴平行，
        # 防推进中姿态漂移导致枪头歪斜蹭碰
        q_des = c.ik.solve(actual, target_rot=cs1_mat, q_init=q_init,
                           site_id=c.gun_site_id, z_align_only=True,
                           retry=False, max_travel=0.1)
        stb["prev_q_des"] = q_des.copy()
        if i % 20 == 0 or f_mag > 10:
            dq = float(np.max(np.abs(q_des - q_cur)))
            depth = float((gun_act - cs1_pos) @ ax_)
            print(f"  [5b step {i}] f={f_mag:.1f} depth={depth*1000:+.1f}mm "
                  f"lat={lat_norm*1000:.1f}mm max|Δq|={dq:.3f}rad")
        return q_des

    def done_b(c):
        # 到位：枪体接触插座底部检测盘 charing_pan
        if c.env.body_collides_with(Task.GUN_BODY, Task.CHARGING_PAN_BODY):
            # 诊断：打印具体接触 geom 对（确认哪个枪部件碰盘）
            m_, d_ = c.env.model, c.env.data
            pan_bid = mujoco.mj_name2id(m_, mujoco.mjtObj.mjOBJ_BODY,
                                        Task.CHARGING_PAN_BODY)
            pan_gs = set(range(m_.body_geomadr[pan_bid],
                               m_.body_geomadr[pan_bid] + m_.body_geomnum[pan_bid]))
            pairs = set()
            for ci in range(d_.ncon):
                ct = d_.contact[ci]
                for a, b in ((ct.geom1, ct.geom2), (ct.geom2, ct.geom1)):
                    if b in pan_gs:
                        na = mujoco.mj_id2name(m_, mujoco.mjtObj.mjOBJ_GEOM, a)
                        if na:
                            pairs.add(na)
            c.phase_msg = (f"插枪到位：charging_gun_1 接触 charing_pan "
                           f"(部件={sorted(pairs)})")
            # 诊断：枪头前端与检测盘的实际几何关系（验证接触位置合理性）
            gun_mat = c.env.site_pose(Task.GUN_SITE)[1]
            gun_pos = c.env.site_pose(Task.GUN_SITE)[0]
            cs1_p, cs1_m = c.env.site_pose(Task.CHARGING_SITE_1)
            ax_w = cs1_m[:, 2]
            z_in = -gun_mat[:, 2] if float(gun_mat[:, 2] @ ax_w) < 0 else gun_mat[:, 2]
            tip = gun_pos + 0.05 * z_in  # col_1 朝内端面中心
            pan_pos = d_.xpos[pan_bid].copy()
            tip_depth = float((tip - cs1_p) @ ax_w)
            tip_lat = np.linalg.norm((tip - cs1_p) - tip_depth * ax_w)
            print(f"  [5b done] 枪头前端={np.round(tip,4)} 深度={tip_depth*1000:+.1f}mm "
                  f"横向={tip_lat*1000:.1f}mm 检测盘={np.round(pan_pos,4)} "
                  f"盘深度={float((pan_pos-cs1_p)@ax_w)*1000:+.1f}mm")
            return True
        # 力安全限位
        f_mag = float(np.linalg.norm(
            c.env.contact_force_between(Task.GUN_BODY, Task.SOCKET_BODY)))
        if f_mag > 80.0:
            c.phase_msg = f"5b 力安全限位 ({f_mag:.1f}N)"
            return True
        # 深度超限仍未接触（charing_pan 在 100mm 深处，枪头前端
        # = gun_site 深度 + 50mm，正常 ~95mm 触盘；超限说明异常）
        cs1_pos, cs1_mat = c.env.site_pose(Task.CHARGING_SITE_1)
        gun_p = c.env.site_pose(Task.GUN_SITE)[0]
        depth = float((gun_p - cs1_pos) @ cs1_mat[:, 2])
        if depth > 0.115:
            c.phase_msg = f"5b 深度超限未接触 charing_pan ({depth*1000:.0f}mm)"
            return True
        return False

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
        Phase(name="phase5b_insert_down",
              desc="沿插座 z+ 插入 0.1m 至接触 charing_pan",
              trajectory=None, n_steps=300, grip_ratio=GRIP_CLOSE,
              on_enter=enter_b, on_step=step_b, on_exit=exit_b,
              done_condition=done_b, monitor_force=True),
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
