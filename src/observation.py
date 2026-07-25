"""观测构建与辅助（当前 obs 在 ArmEnv 内构建，此处放扩展函数）。"""
from __future__ import annotations
import numpy as np


def relative_pose(obs: np.ndarray) -> np.ndarray:
    """从 25 维 obs 提取末端位姿的便捷切片。"""
    qpos = obs[0:6]
    qvel = obs[6:12]
    ee_pos = obs[12:15]
    ee_mat = obs[15:24].reshape(3, 3)
    grip = obs[24]
    return qpos, qvel, ee_pos, ee_mat, grip
