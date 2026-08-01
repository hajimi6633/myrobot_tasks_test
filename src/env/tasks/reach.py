"""到达任务：末端到随机目标点，距离阈值内 +reward，到位 done。"""
from __future__ import annotations
import numpy as np
from src.env.tasks.base import BaseTask


class ReachTask(BaseTask):
    def __init__(self, target_range: float = 0.3, success_thresh: float = 0.02):
        super().__init__()
        self.target_range = target_range
        self.success_thresh = success_thresh
        self.target = np.zeros(3)

    def reset(self, seed: int | None = None):
        super().reset(seed)
        # 在工作台上方随机采点（ee_link 现在位于两手指中点，
        # z 范围适配新末端工作空间，避开肘部奇异区）
        self.target = np.array([
            self.rng.uniform(-0.3, 0.3),
            self.rng.uniform(0.1, 0.6),
            self.rng.uniform(0.65, 0.95),
        ])
        # 红色标记点：在 target 位置画红球，便于在 viewer 中观察目标
        self.markers = [{"pos": self.target.copy(), "rgba": [1, 0, 0, 1], "size": 0.02}]
        return {"target": self.target.copy()}

    def compute_reward(self, obs: np.ndarray):
        d = float(np.linalg.norm(self.ee_pos - self.target))
        reward = -d
        done = d < self.success_thresh
        info = {"dist": d, "target": self.target.copy()}
        if done:
            reward += 1.0
        return reward, done, info
