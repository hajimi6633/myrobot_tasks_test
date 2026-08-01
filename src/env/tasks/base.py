"""任务基类：任务通过 attach/env 注入，提供 reset/step 接口。

任务负责定义目标、奖励、done 条件，env 负责物理推进。
子类实现 reset_task / compute_reward。
"""
from __future__ import annotations
import numpy as np


class BaseTask:
    def __init__(self):
        self.env = None
        # 可视化标记：list of dict{pos, rgba, size}，由子类在 reset 中填充，
        # env.render_markers(viewer) 会在 viewer 中绘制（不影响物理）
        self.markers: list[dict] = []

    def attach(self, env):
        self.env = env

    def reset(self, seed: int | None = None):
        """每局开始时采样目标/物体位姿。子类可覆写。"""
        self.rng = np.random.default_rng(seed)
        self.markers = []

    def step(self, obs: np.ndarray) -> tuple[float, bool, dict]:
        return self.compute_reward(obs)

    def compute_reward(self, obs: np.ndarray) -> tuple[float, bool, dict]:
        raise NotImplementedError

    # ---- 便捷访问 ----
    @property
    def ee_pos(self) -> np.ndarray:
        """末端位置 [3]，来源由 env.pose_provider 决定（GT 或视觉）。"""
        return self.env.ee_pose()[0]

    def body_pos(self, name: str) -> np.ndarray:
        """指定 body 位置 [3]，来源由 env.pose_provider 决定。"""
        return self.env.body_pose(name)[0]
