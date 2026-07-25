"""批量评估任务平均回报与成功率。

用法: python -m scripts.eval --task reach --episodes 20
"""
from __future__ import annotations
import argparse
import numpy as np

from src.env.arm_env import ArmEnv
from src.control.ik_solver import IKSolver
from src.control.trajectory import min_jerk, Trajectory
from scripts.run_task import TASKS, plan_to_target


def run_episode(env, ik, task, max_steps=300, dt_ctrl=0.05, T=2.0):
    obs = env.reset()
    target = getattr(task, "target", None)
    if target is None:
        target = getattr(task, "place_target", None)
    traj = plan_to_target(env, ik, target, dt_ctrl=dt_ctrl, T=T) if target is not None else None
    idx = 0
    total_r, done = 0.0, False
    for _ in range(max_steps):
        q_des = traj.at(idx) if (traj is not None and idx < len(traj)) else env.data.qpos[env.arm_qposadr].copy()
        if traj is not None: idx += 1
        res = env.step(np.concatenate([q_des, [1.0]]))
        total_r += res.reward
        if res.done:
            done = True
            break
    return total_r, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="reach")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--mode", choices=["position", "torque"], default="position")
    args = ap.parse_args()

    env = ArmEnv(action_mode=args.mode, task=TASKS[args.task]())
    ik = IKSolver(env)
    rewards, succ = [], []
    for ep in range(args.episodes):
        r, done = run_episode(env, ik, env.task)
        rewards.append(r); succ.append(done)
        print(f"ep{ep:02d}  reward={r:7.3f}  success={done}")
    print(f"\nmean_reward={np.mean(rewards):.3f}  success_rate={np.mean(succ):.2%}")


if __name__ == "__main__":
    main()
