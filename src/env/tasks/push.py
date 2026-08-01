"""推物任务：用末端把桌面物体推到目标线/点，接触丰富任务。

奖励 = 物体靠近目标 - 接触前的远距离惩罚，强调力控平稳。
"""
from __future__ import annotations
import numpy as np
import mujoco
from src.env.tasks.base import BaseTask


class PushTask(BaseTask):
    def __init__(self, success_thresh: float = 0.03):
        super().__init__()
        self.success_thresh = success_thresh
        self.object_name = "object"
        self.target = np.array([-0.3, 0.5, 0.43])

    def reset(self, seed: int | None = None):
        super().reset(seed)
        oid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, self.object_name)
        self.env.data.qpos[self.env.model.jnt_qposadr[
            self.env.model.body_jntadr[oid]]] = 0.2
        self.env.data.qpos[self.env.model.jnt_qposadr[
            self.env.model.body_jntadr[oid] + 1]] = 0.5
        mujoco.mj_forward(self.env.model, self.env.data)
        return {"target": self.target.copy()}

    def _object_pos(self) -> np.ndarray:
        """物体位置，来源由 env.pose_provider 决定（GT 或视觉）。"""
        return self.env.body_pose(self.object_name)[0]

    def compute_reward(self, obs: np.ndarray):
        obj = self._object_pos()
        d = float(np.linalg.norm(obj[:2] - self.target[:2]))  # 桌面平面距离
        ee_d = float(np.linalg.norm(self.ee_pos - obj))
        # 推动方向：物体在 ee 与目标连线之间更优
        reward = -d - 0.1 * ee_d
        done = d < self.success_thresh
        info = {"dist": d}
        if done:
            reward += 2.0
        return reward, done, info
