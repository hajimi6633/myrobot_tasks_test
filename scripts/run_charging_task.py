"""执行充电枪抓取-插拔长序列任务（6 阶段）。

用法:
  python -m scripts.run_charging_task --viewer
  python -m scripts.run_charging_task            # 无 viewer
"""
from __future__ import annotations
import argparse
import numpy as np
import mujoco

from src.env.arm_env import ArmEnv
from src.control.ik_solver import IKSolver
from src.control.force_sensor import ForceTorqueSensor
from src.control.constraints import ConstraintManager
from src.control.grasp import GraspCoupler
from src.control.impedance import AdmittanceController
from src.env.tasks.charging_gun import ChargingGunTask, PhaseContext
from src.env.tasks.charging_phases import build_all_phases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--max-steps", type=int, default=4000)
    args = ap.parse_args()

    task = ChargingGunTask()
    env = ArmEnv(action_mode="position", task=task, gravity_comp=True)
    ik = IKSolver(env)
    obs = env.reset(seed=0)

    # 基础设施
    force_sensor = ForceTorqueSensor(env.model, env.data, ChargingGunTask.GUN_SITE)
    constraints = ConstraintManager(env.model, env.data)
    coupler = GraspCoupler(env.model, env.data,
                           ChargingGunTask.GUN_BODY, ChargingGunTask.GRIPPER_BASE_BODY)
    admittance = AdmittanceController(mass=1.0, stiffness=150.0,
                                      damping_ratio=1.0, max_delta=0.05)
    ctrl_dt = env.n_substeps * env.model.opt.timestep

    ctx = PhaseContext(env=env, ik=ik, task=task, force_sensor=force_sensor,
                       constraints=constraints, coupler=coupler,
                       admittance=admittance, dt_ctrl=ctrl_dt)
    # IK 直接对 gun_site 求解时需同步更新耦合体（枪 freejoint 跟随臂）
    ik.coupler = coupler
    ctx.gun_site_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_SITE, ChargingGunTask.GUN_SITE)

    # 确保 eq_socgun_1 初始激活（枪固定在插座）
    constraints.set_active(ChargingGunTask.EQ_SOCKET, True)
    mujoco.mj_forward(env.model, env.data)

    phases = build_all_phases(ctx, env.home_qpos)

    viewer = env.launch_passive_viewer() if args.viewer else None
    if viewer is not None:
        print("Passive viewer 已启动。Ctrl+C 退出。")

    step = 0
    aborted = False
    for pi, phase in enumerate(phases):
        ctx.phase_done = False
        ctx.phase_msg = ""
        ctx.forces = []
        print(f"\n[phase {pi}] {phase.name}: {phase.desc} ({phase.length} 步, "
              f"夹爪{'张开' if phase.grip_ratio > 0.5 else '闭合'})")
        if phase.on_enter:
            phase.on_enter(ctx)
        if ctx.phase_done:
            print(f"  跳过: {ctx.phase_msg}")
            if "失败" in ctx.phase_msg:
                aborted = True
                break
            continue

        for i in range(phase.length):
            if step >= args.max_steps:
                break
            q_des = phase.on_step(ctx, i)
            action = np.concatenate([q_des, [phase.grip_ratio]])
            env.step(action)
            step += 1
            if phase.monitor_force:
                # 导纳段记录枪与环境的接触力，非导纳段记录力传感器读数
                if phase.use_admittance:
                    # 插枪段记录枪-车插座力，归位段记录枪-充电插座力
                    other = (ChargingGunTask.CAR_SOCKET_BODY
                             if "phase3" in phase.name else ChargingGunTask.SOCKET_BODY)
                    fval = float(np.linalg.norm(
                        env.contact_force_between(ChargingGunTask.GUN_BODY, other)))
                else:
                    fval = force_sensor.force_magnitude()
                ctx.forces.append(fval)
            if viewer is not None:
                env.render_markers(viewer)
                viewer.sync()
            if phase.done_condition is not None and phase.done_condition(ctx):
                print(f"  提前完成: {ctx.phase_msg}")
                break
            if ctx.phase_done:
                break

        if phase.on_exit:
            phase.on_exit(ctx)

        # 诊断：仿真实际 ee_link 位置 vs 目标 site 位置（跟踪误差）
        # 对比 IK 理论误差（on_enter 打印的 [IK] 误差）与仿真实际误差，
        # 可区分"IK 解不准" vs "position controller 跟踪不到位"
        mujoco.mj_forward(env.model, env.data)
        ee_actual = env.data.site_xpos[env.ee_site_id].copy()
        q_actual = env.data.qpos[env.arm_qposadr].copy()
        q_goal_final = phase.on_step(ctx, phase.length - 1) if phase.on_step else None
        # 尝试从阶段名解析目标 site（1a_to_gun_site_2 → gun_site_2）
        target_site_name = None
        for sname in [ChargingGunTask.GUN_SITE_2, ChargingGunTask.GUN_SITE_1,
                      ChargingGunTask.CHARGING_SITE_2, ChargingGunTask.CAR_SITE_2,
                      ChargingGunTask.CHARGING_SITE_1, ChargingGunTask.GUN_SITE]:
            if sname in phase.name:
                target_site_name = sname
                break
        if target_site_name is not None:
            tgt_pos, tgt_mat = env.site_pose(target_site_name)
            tgt_label = target_site_name
            # 预接近段（back_along_z）实际 IK 目标带偏移，用规划目标对比才有意义
            if getattr(ctx, "last_plan_name", "") == phase.name and \
                    hasattr(ctx, "last_ik_target"):
                tgt_pos = ctx.last_ik_target.copy()
                tgt_label = f"{target_site_name}+offset"
            track_err = np.linalg.norm(ee_actual - tgt_pos)
            # 姿态误差：ee_link z 轴与目标 site z 轴夹角 (deg)
            ee_mat_now = env.data.site_xmat[env.ee_site_id].reshape(3, 3)
            ee_z = ee_mat_now[:, 2]
            tgt_z = tgt_mat[:, 2]
            cos_a = float(np.clip(np.dot(ee_z, tgt_z), -1.0, 1.0))
            z_angle = float(np.degrees(np.arccos(abs(cos_a))))
            # 全姿态误差：R_err 转角 (deg)，0=三轴完全一致
            R_err = tgt_mat @ ee_mat_now.T
            cos_theta = float(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
            rot_angle = float(np.degrees(np.arccos(abs(cos_theta))))
            # 关节角跟踪误差（qpos 是否到达 q_goal）
            if q_goal_final is not None:
                q_track_err = np.max(np.abs(q_actual - q_goal_final))
                # 进一步诊断：把 q_goal_final 赋给 qpos 后 forward，看 ee 理论位置
                # 与仿真实际 qpos 对应的 ee 位置差多少（排除 IK 求解 vs 跟踪问题）
                backup_q = env.data.qpos.copy()
                env.data.qpos[env.arm_qposadr] = q_goal_final
                mujoco.mj_forward(env.model, env.data)
                ee_at_qgoal = env.data.site_xpos[env.ee_site_id].copy()
                env.data.qpos[:] = backup_q
                mujoco.mj_forward(env.model, env.data)
                ik_err = np.linalg.norm(ee_at_qgoal - tgt_pos)
                print(f"  [实际] ee={np.round(ee_actual,4)} 目标({tgt_label})={np.round(tgt_pos,4)} "
                      f"位置误差={track_err:.4f}m z轴夹角={z_angle:.1f}deg "
                      f"全姿态偏差={rot_angle:.1f}deg "
                      f"关节跟踪max|Δq|={q_track_err:.4f}rad")
                print(f"  [理论] q_goal对应ee={np.round(ee_at_qgoal,4)} "
                      f"IK理论误差={ik_err:.4f}m (q_actual对应ee={np.round(ee_actual,4)})")
                # 一致性校验：轨迹终点应等于 IK 解（settle 段保证）
                last_ik_q = getattr(ctx, "last_ik_q", None)
                if last_ik_q is not None:
                    q_consist = np.max(np.abs(q_goal_final - last_ik_q))
                    print(f"  [校验] 轨迹终点 vs IK解 max|Δq|={q_consist:.6f}rad "
                          f"(>0 表示轨迹未以 IK 解结尾)")
                # 逐关节诊断：跟踪误差 + 执行器力 vs 力限（暴露力饱和）
                qvel_norm = float(np.linalg.norm(
                    env.data.qvel[env.arm_dofadr]))
                print(f"  [关节] |qvel|={qvel_norm:.3f}rad/s "
                      f"(<0.05=已静止, >0.3=仍在运动)")
                for k, jname in enumerate(env.arm_joint_names):
                    aid = env.act_act_ids[jname]
                    f_now = float(env.data.actuator_force[aid])
                    f_lim = float(env.model.actuator_forcerange[aid][1])
                    # 约束力（接触/限位/等式约束）：与执行器力对顶说明被约束卡住
                    f_con = float(env.data.qfrc_constraint[env.arm_dofadr[k]])
                    ej = q_actual[k] - q_goal_final[k]
                    sat = "  <== 力饱和!" if abs(f_now) >= 0.95 * f_lim else ""
                    con = f"  <== 被约束顶住({f_con:+.1f}N·m)!" if abs(f_con) > 5.0 else ""
                    print(f"    {jname}: q={q_actual[k]:+.3f} "
                          f"des={q_goal_final[k]:+.3f} err={ej:+.3f} "
                          f"F={f_now:+.1f}/{f_lim:.0f}N·m{sat}{con}")
            else:
                print(f"  [实际] ee={np.round(ee_actual,4)} 目标site={np.round(tgt_pos,4)} "
                      f"位置误差={track_err:.4f}m z轴夹角={z_angle:.1f}deg "
                      f"全姿态偏差={rot_angle:.1f}deg")

        # 碰撞诊断：任何涉及臂链 body 的接触对（含臂自碰撞、地板、枪、插座）
        # 用于区分"控制器跟踪滞后"与"物理卡住（碰撞）"
        arm_bodies = {"shoulder_Link", "upper_arm_Lift", "forearm_Link",
                      "wrist_1_Link", "wrist_2_Link", "wrist_3_Link",
                      "ee_off", "gripper_base", "finger_left", "finger_right"}
        hits = {}
        for ci in range(env.data.ncon):
            c = env.data.contact[ci]
            b1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY,
                                   env.model.geom_bodyid[c.geom1])
            b2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY,
                                   env.model.geom_bodyid[c.geom2])
            if b1 in arm_bodies or b2 in arm_bodies:
                key = f"{b1} <-> {b2}"
                hits[key] = hits.get(key, 0) + 1
        if hits:
            print(f"  [碰撞] 涉及臂体的接触: {hits}")

        # 力监测摘要
        if phase.monitor_force and ctx.forces:
            print(f"  力: 峰值 {max(ctx.forces):.2f}N, 末值 {ctx.forces[-1]:.2f}N")
        if ctx.phase_msg:
            print(f"  {ctx.phase_msg}")
        # 诊断：打印 gun-ee 局部系偏移（抓取后应恒定，漂移说明耦合失效）
        if coupler.attached:
            mujoco.mj_forward(env.model, env.data)
            gun_p = env.site_pose(ChargingGunTask.GUN_SITE)[0]
            ee_p = env.data.site_xpos[env.ee_site_id].copy()
            ee_mat = env.data.site_xmat[env.ee_site_id].reshape(3, 3)
            dyn_off_local = ee_mat.T @ (gun_p - ee_p)
            drift = np.linalg.norm(dyn_off_local - ctx.ee_to_gun_offset)
            print(f"  [diag] gun={np.round(gun_p,4)} ee={np.round(ee_p,4)} "
                  f"局部偏移={np.round(dyn_off_local,4)} "
                  f"漂移={drift:.4f}m")
        if ctx.phase_done and "失败" in ctx.phase_msg:
            aborted = True
            break
        if step >= args.max_steps:
            print("达到最大步数，终止")
            break

    print(f"\n{'任务中止' if aborted else '任务完成'}，共 {step} 步")
    if viewer is not None:
        print("关闭 viewer 窗口退出。")
        try:
            while viewer.is_running():
                env.render_markers(viewer)
                viewer.sync()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
