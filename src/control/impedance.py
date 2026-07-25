"""笛卡尔阻抗控制（接触任务力位混合占位）。

简化实现：在末端建立弹簧-阻尼模型，把期望的笛卡尔误差映射到关节力矩。
完整版需考虑重力补偿与惯量加权，这里给出可跑通的骨架。
"""
from __future__ import annotations
import numpy as np
import mujoco


class CartesianImpedance:
    def __init__(self, env, kp_cart: float = 200.0, kv_cart: float = 20.0,
                 max_torque: float = 150.0):
        self.env = env
        self.kp = kp_cart
        self.kv = kv_cart
        self.max_torque = max_torque

    def compute(self, pos_des: np.ndarray, pos_cur: np.ndarray,
                vel_des: np.ndarray | None = None, vel_cur: np.ndarray | None = None
                ) -> np.ndarray:
        m, d = self.env.model, self.env.data
        vel_des = np.zeros(3) if vel_des is None else vel_des
        vel_cur = np.zeros(3) if vel_cur is None else vel_cur

        # 末端力 = Kp*(x_d - x) + Kv*(xd_d - xd)
        force = self.kp * (pos_des - pos_cur) + self.kv * (vel_des - vel_cur)

        jacp = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, None, self.env.ee_site_id)
        jac = jacp[:, self.env.arm_dofadr]
        # τ = J^T F
        tau = jac.T @ force
        return np.clip(tau, -self.max_torque, self.max_torque)
