"""StackEnv：把节点栈包装成 gymnasium.Env（保 eval.py 的 RL 用途）。

obs 聚合自 SimNode.joint_state + PoseNode；reward / done 由任务层提供。
未安装 gymnasium（可选依赖）时退化为裸接口，仍可 reset/step。
"""
from __future__ import annotations
import numpy as np

try:
    import gymnasium as gym
    _Base = gym.Env
except ImportError:                    # rl 是可选依赖（pyproject [rl]）
    _Base = object


class StackEnv(_Base):
    def __init__(self, scene_xml: str):
        from rcs.launch import build_charging_stack
        self.ex, self.h = build_charging_stack(scene_xml)
        self.sim = self.h["sim"]
        # gymnasium 空间（安装了才暴露）
        if hasattr(self, "action_space"):
            self.observation_space = None     # TODO: 按需补 spaces.Box
            self.action_space = None

    @property
    def obs_dim(self) -> int:
        # 6 qpos + 6 qvel + 3 pos + 9 rot + 1 grip（与旧 ArmEnv 一致）
        return 25

    def _obs(self) -> np.ndarray:
        q, dq = self.sim.joint_state()
        pos, rot = self.h["pose"]._get_ee_pose()
        # TODO 迁移：gripper opening（旧 ArmEnv._build_obs 的最后一位）
        return np.concatenate([q, dq, pos, rot.flatten(), [1.0]]).astype(np.float32)

    def reset(self, seed=None):
        self.sim.reset()
        self.ex.spin_once()             # 推一拍发布初始状态（含快照）
        return self._obs(), {}

    def step(self, action):
        # action = [q_arm(6), grip(1)]：作为期望指令经安全闸门下发
        a = np.asarray(action, float).reshape(-1)
        self.h["ctrl"]._on_qdes((a[:6], float(a[6])), None)
        self.ex.spin_once()
        # TODO 迁移：reward / done 接任务层（旧 task.compute_reward）
        return self._obs(), 0.0, False, {}
