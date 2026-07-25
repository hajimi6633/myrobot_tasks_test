"""跑指定任务：用 IK + 轨迹规划闭环演示。

用法:
  python -m scripts.run_task --task reach --steps 300 --mode position
  python -m scripts.run_task --task reach --viewer   # 带 viewer
  python -m scripts.run_task --task pick_place --viewer
"""
from __future__ import annotations
import argparse
import numpy as np
import mujoco

from src.env.arm_env import ArmEnv
from src.control.ik_solver import IKSolver
from src.control.trajectory import min_jerk, Trajectory
from src.env.tasks.reach import ReachTask
from src.env.tasks.pick_place import PickPlaceTask
from src.env.tasks.push import PushTask

TASKS = {"reach": ReachTask, "pick_place": PickPlaceTask, "push": PushTask}


def plan_to_target(env, ik, target_pos, dt_ctrl=0.05, T=2.0,
                   q_init=None, target_rot=None):
    """用 IK 解到目标，生成 min-jerk 轨迹。返回 (Trajectory, q_goal)。

    target_rot 非 None 时约束末端姿态。q_goal 供下一段轨迹衔接使用。
    """
    q_cur = q_init if q_init is not None else env.data.qpos[env.arm_qposadr].copy()
    q_goal = ik.solve(target_pos, target_rot=target_rot, q_init=q_cur)
    wp = min_jerk(q_cur, q_goal, T, dt_ctrl)
    return Trajectory(wp, dt_ctrl), q_goal


def make_hold_traj(q, n_steps, dt_ctrl=0.05):
    """生成保持指定关节角的轨迹（n_steps 个相同点），用于夹爪开合等待段。"""
    wp = np.tile(np.asarray(q, float), (n_steps, 1))
    return Trajectory(wp, dt_ctrl)


def check_grasp(env) -> float:
    """返回当前夹爪接触力 (N)，用于抓取检测。"""
    return env.gripper.get_contact_force()


def build_reach_plan(env, ik, task, dt_ctrl=0.05):
    """reach: 单段轨迹到 target，夹爪张开。返回 [(traj, grip, monitor)]。"""
    traj, _ = plan_to_target(env, ik, task.target, dt_ctrl=dt_ctrl, T=2.0)
    return [(traj, 1.0, False)]


def build_pick_place_plan(env, ik, task, dt_ctrl=0.05):
    """pick_place 六段：approach → grasp → 闭合夹爪 → lift → place → 张开夹爪。

    夹爪开合单独成 hold 段（臂保持不动），保证下一段运动前夹爪已完全到位。
    抓取姿态约束：ee_link 的 Z 轴与世界系 -Z（桌面法向反向，朝下）对齐，
    即从 home 位姿读取 ee_link 旋转矩阵作为目标姿态，全程保持垂直抓取。
    每段轨迹以上一段终点 q_goal 作为 IK 初值，保证段间状态连续衔接。
    闭合后和 lift 段做抓取检测（基于夹爪接触力）。

    返回 [(Trajectory, grip_ratio, monitor), ...]。
    grip_ratio: 1.0=张开, 0.0=闭合
    monitor: True=该段记录夹爪力并在段末判定抓取状态
    """
    # 取 home 位姿下 ee_link 的旋转矩阵作为"垂直抓取姿态"参考
    mujoco.mj_forward(env.model, env.data)
    grasp_rot = env.data.site_xmat[env.ee_site_id].reshape(3, 3).copy()

    obj = task.object_init_pos
    place = task.place_target
    approach_h = 0.08   # approach 起点在物体上方 8cm
    lift_h = 0.15       # 抬升到物体上方 15cm 再平移
    grasp_wait_s = 0.5  # 夹爪开合等待时长 (s)
    grasp_wait_n = int(grasp_wait_s / dt_ctrl)

    # 1) approach: 物体正上方，张开夹爪准备抓取
    p1 = obj.copy(); p1[2] += approach_h
    # 2) grasp: 下降到物体位置，仍张开（到位后再闭合）
    p2 = obj.copy()
    # 3) lift: 抬升到物体上方，闭合夹爪（抓起）
    p3 = obj.copy(); p3[2] += lift_h
    # 4) place: 平移到 place_target，闭合夹爪
    p4 = place.copy()

    segs = []
    q_cur = None  # 第一段用 env 当前状态（home）作为起点
    # 1) approach 张开
    traj, q_goal = plan_to_target(env, ik, p1, dt_ctrl=dt_ctrl, T=1.5,
                                   q_init=q_cur, target_rot=grasp_rot)
    segs.append((traj, 1.0, False))
    q_cur = q_goal
    # 2) grasp 下降 张开
    traj, q_goal = plan_to_target(env, ik, p2, dt_ctrl=dt_ctrl, T=1.0,
                                   q_init=q_cur, target_rot=grasp_rot)
    segs.append((traj, 1.0, False))
    q_cur = q_goal
    # 3) hold 闭合夹爪（臂保持不动，等夹爪闭合完成）+ 抓取检测
    segs.append((make_hold_traj(q_cur, grasp_wait_n, dt_ctrl), 0.0, True))
    # 4) lift 闭合 + 抓取检测（运动中若脱离会检出）
    traj, q_goal = plan_to_target(env, ik, p3, dt_ctrl=dt_ctrl, T=1.0,
                                   q_init=q_cur, target_rot=grasp_rot)
    segs.append((traj, 0.0, True))
    q_cur = q_goal
    # 5) place 闭合
    traj, q_goal = plan_to_target(env, ik, p4, dt_ctrl=dt_ctrl, T=2.0,
                                   q_init=q_cur, target_rot=grasp_rot)
    segs.append((traj, 0.0, False))
    q_cur = q_goal
    # 6) hold 张开夹爪（松开物体）
    segs.append((make_hold_traj(q_cur, grasp_wait_n, dt_ctrl), 1.0, False))
    return segs


def build_push_plan(env, ik, task, dt_ctrl=0.05):
    """push: 默认单段到目标，夹爪闭合（用夹爪侧面推）。"""
    traj, _ = plan_to_target(env, ik, task.target, dt_ctrl=dt_ctrl, T=2.0)
    return [(traj, 0.0, False)]


PLAN_BUILDERS = {
    "reach": build_reach_plan,
    "pick_place": build_pick_place_plan,
    "push": build_push_plan,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="reach")
    ap.add_argument("--mode", choices=["position"], default="position",
                    help="控制模式（torque 暂未启用）")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    task = TASKS[args.task]()
    env = ArmEnv(action_mode=args.mode, task=task, gravity_comp=True)
    ik = IKSolver(env)
    obs = env.reset(seed=0)

    viewer = env.launch_passive_viewer() if args.viewer else None
    if viewer is not None:
        print("Passive viewer 已启动，可实时观察。Ctrl+C 退出。")

    # 构建多段轨迹 + 每段夹爪开合
    segments = PLAN_BUILDERS[args.task](env, ik, task)

    total_r = 0.0
    step = 0
    res = None
    # 抓取检测阈值（可按物体/夹爪调整）
    GRASP_FORCE_MIN = 1.0      # 最小抓住力 (N)，低于此视为未抓住
    DROP_RATIO = 0.3           # 力降至峰值的此比例以下视为脱离

    for seg_idx, seg in enumerate(segments):
        traj, grip_ratio = seg[0], seg[1]
        monitor = seg[2] if len(seg) > 2 else False
        forces = []  # monitor 时记录每步夹爪力
        print(f"[seg {seg_idx}] 段长 {len(traj)} 步, 夹爪 {'张开' if grip_ratio > 0.5 else '闭合'}"
              f"{', 监测抓取' if monitor else ''}")
        for traj_idx in range(len(traj)):
            if step >= args.steps:
                break
            q_des = traj.at(traj_idx)
            action = np.concatenate([q_des, [grip_ratio]])
            res = env.step(action)
            total_r += res.reward
            step += 1
            if monitor:
                forces.append(check_grasp(env))
            if viewer is not None:
                env.render_markers(viewer)
                viewer.sync()
            if res.done:
                print(f"[step {step-1}] done, reward={res.reward:.3f}")
                break
        # 段末抓取状态分析
        if monitor and not res.done:
            cur_force = forces[-1] if forces else 0.0
            max_force = max(forces) if forces else 0.0
            if max_force < GRASP_FORCE_MIN:
                print(f"[seg {seg_idx}] 抓取失败: 夹爪无力 (max={max_force:.2f}N < {GRASP_FORCE_MIN}N), 物体未抓住")
                break
            if cur_force < DROP_RATIO * max_force:
                print(f"[seg {seg_idx}] 物体脱离: 夹爪力骤减 {max_force:.2f}N -> {cur_force:.2f}N")
                break
            print(f"[seg {seg_idx}] 抓取稳定: 当前力 {cur_force:.2f}N (峰值 {max_force:.2f}N)")
        if res.done:
            break

    print(f"total_reward={total_r:.3f} over {step} steps")


if __name__ == "__main__":
    main()
