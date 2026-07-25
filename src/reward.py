"""奖励函数工具（任务自带 reward，此处放通用 shaping）。"""
from __future__ import annotations
import numpy as np


def distance_shaping(cur_pos: np.ndarray, prev_pos: np.ndarray,
                     target: np.ndarray, beta: float = 1.0) -> float:
    """势能塑形：奖励接近目标的位移变化。"""
    d_cur = np.linalg.norm(cur_pos - target)
    d_prev = np.linalg.norm(prev_pos - target)
    return beta * (d_prev - d_cur)
