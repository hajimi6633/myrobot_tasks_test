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
