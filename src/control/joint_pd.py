"""关节空间 PD 控制器（torque 模式下手动计算力矩）。"""
from __future__ import annotations
import numpy as np


class JointPD:
    def __init__(self, kp: np.ndarray, kv: np.ndarray,
                 max_torque: float = 150.0):
        self.kp = np.asarray(kp, float)
        self.kv = np.asarray(kv, float)
        self.max_torque = max_torque

    def compute(self, q_des: np.ndarray, q: np.ndarray,
                qd_des: np.ndarray | None = None, qd: np.ndarray | None = None) -> np.ndarray:
        qd_des = np.zeros_like(q_des) if qd_des is None else qd_des
        qd = np.zeros_like(q) if qd is None else qd
        tau = self.kp * (q_des - q) + self.kv * (qd_des - qd)
        return np.clip(tau, -self.max_torque, self.max_torque)
