"""导纳控制：读末端外力 → 阻抗模型积分 → 位置修正 → 喂给 IK。

导纳控制适合 position 控制底层：外层读力，按二阶阻抗模型
    M·δẍ + B·δẋ + K·δx = F_ext
积分得到顺从偏移 δx，实际目标位置 = 名义目标 + δx。
外力越大退让越多，实现接触柔顺（插枪遇阻调整）。

临界阻尼: B = 2*sqrt(M*K)。
"""
from __future__ import annotations
import numpy as np


class AdmittanceController:
    """笛卡尔导纳控制器（仅位置，3 自由度）。"""

    def __init__(self, mass: float = 1.0, stiffness: float = 200.0,
                 damping_ratio: float = 1.0, max_delta: float = 0.03):
        """
        mass: 虚拟惯量 M
        stiffness: 虚拟刚度 K
        damping_ratio: 阻尼比 ζ，1=临界阻尼（不振荡）
        max_delta: 单轴最大顺从偏移 (m)，防止退让过大
        """
        self.M = float(mass)
        self.K = float(stiffness)
        self.B = 2.0 * damping_ratio * np.sqrt(self.M * self.K)
        self.max_delta = float(max_delta)
        self.delta = np.zeros(3)        # 顺从偏移
        self.delta_dot = np.zeros(3)    # 偏移速度

    def reset(self):
        self.delta[:] = 0.0
        self.delta_dot[:] = 0.0

    def step(self, f_ext: np.ndarray, dt: float) -> np.ndarray:
        """一步积分，返回当前顺从偏移 δx。

        f_ext: 世界系末端外力 [fx,fy,fz]（N），来自力传感器
        dt: 控制周期 (s)

        内置两级稳定措施（碰撞冲击下防发散）：
        1. 力输入限幅：冲击力截断到 F_CLIP，避免瞬时大力把 delta_dot 打飞
        2. delta 限速：每步偏移增量截断，保证输出连续平滑
        """
        f = np.asarray(f_ext, dtype=float).reshape(3)
        # 力输入限幅：碰撞瞬间的冲击力远超稳态接触力，直接截断
        F_CLIP = 30.0
        f = np.clip(f, -F_CLIP, F_CLIP)
        # 增量限速的基准必须在积分【前】记录——若积分后再取，
        # clip 相对的是跳变后的值，限速完全失效，碰撞瞬间 delta
        # 会被 max_delta 一步拉满（目标跳变 → q_des 突变）
        delta_prev = self.delta.copy()
        # M·δẍ = F_ext - B·δẋ - K·δx
        delta_ddot = (f - self.B * self.delta_dot - self.K * self.delta) / self.M
        self.delta_dot += delta_ddot * dt
        # 速度限幅：碰撞冲击积分出的 delta_dot 残余大，会让 delta
        # 长期顶满 STEP_CLIP 单向漂移；截到与步增量匹配的速度上限
        V_CLIP = 0.02  # m/s（0.5mm/步的 2 倍裕量，防饱和漂移）
        self.delta_dot = np.clip(self.delta_dot, -V_CLIP, V_CLIP)
        self.delta += self.delta_dot * dt
        # 偏移增量限速（每步每轴最多 STEP_CLIP），保证输出连续平滑
        STEP_CLIP = 0.0005  # 0.5mm/步 @50Hz = 25mm/s 上限
        self.delta = np.clip(self.delta, delta_prev - STEP_CLIP,
                             delta_prev + STEP_CLIP)
        # 总偏移限幅
        self.delta = np.clip(self.delta, -self.max_delta, self.max_delta)
        return self.delta.copy()

    def correct(self, x_nominal: np.ndarray) -> np.ndarray:
        """名义目标位置 + 顺从偏移 = 实际目标位置。"""
        return np.asarray(x_nominal, dtype=float).reshape(3) + self.delta


class CartesianImpedance:
    """保留旧的力矩式阻抗接口（占位，当前 position 模式下未使用）。"""
    def __init__(self, env, kp_cart: float = 200.0, kv_cart: float = 20.0,
                 max_torque: float = 150.0):
        self.env = env
        self.kp = kp_cart
        self.kv = kv_cart
        self.max_torque = max_torque

    def compute(self, pos_des, pos_cur, vel_des=None, vel_cur=None):
        import mujoco
        m, d = self.env.model, self.env.data
        vel_des = np.zeros(3) if vel_des is None else vel_des
        vel_cur = np.zeros(3) if vel_cur is None else vel_cur
        force = self.kp * (np.asarray(pos_des) - np.asarray(pos_cur)) + self.kv * (vel_des - vel_cur)
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, None, self.env.ee_site_id)
        jac = jacp[:, self.env.arm_dofadr]
        tau = jac.T @ force
        return np.clip(tau, -self.max_torque, self.max_torque)
