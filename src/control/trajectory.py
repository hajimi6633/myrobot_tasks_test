"""轨迹生成：多项式插值 / 梯形速度 / 最小急动度占位。"""
from __future__ import annotations
import numpy as np


def min_jerk(q_start: np.ndarray, q_end: np.ndarray, T: float, dt: float) -> np.ndarray:
    """最小急动度轨迹，返回 (N, dof)。位置 5 次多项式，起止速度加速度为 0。"""
    N = int(T / dt)
    t = np.linspace(0, 1, N)
    tau = 10 * t**3 - 15 * t**4 + 6 * t**5  # 归一化 s(t)
    q_start = np.asarray(q_start, float); q_end = np.asarray(q_end, float)
    return q_start[None, :] + (q_end - q_start)[None, :] * tau[:, None]


def cubic_with_vel(q0, v0, qf, vf, T: float, dt: float) -> np.ndarray:
    """三次多项式，带起末速度。"""
    N = int(T / dt)
    t = np.linspace(0, T, N)
    a = 2*q0 - 2*qf + v0*T + vf*T
    b = -3*q0 + 3*qf - 2*v0*T - vf*T
    q0, v0, qf, vf = map(np.asarray, (q0, v0, qf, vf))
    s = (a * (t**3 / T**3)[:, None]
         + b * (t**2 / T**2)[:, None]
         + v0[None, :] * t[:, None]
         + q0[None, :])
    return s


class Trajectory:
    """通用轨迹容器：持有关节角序列，按时间索引取值。"""

    def __init__(self, waypoints: np.ndarray, dt: float):
        self.wp = np.asarray(waypoints, float)  # (N, dof)
        self.dt = dt

    def __len__(self):
        return len(self.wp)

    def at(self, i: int) -> np.ndarray:
        i = max(0, min(i, len(self.wp) - 1))
        return self.wp[i].copy()

    def __iter__(self):
        for q in self.wp:
            yield q.copy()
