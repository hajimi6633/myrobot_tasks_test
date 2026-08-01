"""抓取放置任务：抓起桌面物体放到目标区域。

阶段机：approach -> grasp -> lift -> place -> done
"""
from __future__ import annotations
import numpy as np
import mujoco
from src.env.tasks.base import BaseTask


class PickPlaceTask(BaseTask):
    def __init__(self, success_thresh: float = 0.04):
        super().__init__()
        self.success_thresh = success_thresh
        self.object_name = "object"
        # place_target 的 z 在 reset 时设为物体初始 z（桌面+物体半高），
        # 保证放置目标在桌面上而非空中
        self.place_target = np.array([0.3, 0.5, 0.0])

    def reset(self, seed: int | None = None):
        super().reset(seed)
        # 物体随机放置在桌面（object 为 freejoint，qpos 连续 7 项：3 平动 + 4 四元数）
        oid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, self.object_name)
        qposadr = self.env.model.jnt_qposadr[self.env.model.body_jntadr[oid]]
        x = self.rng.uniform(-0.2, 0.2)
        y = self.rng.uniform(0.3, 0.7)
        self.env.data.qpos[qposadr] = x
        self.env.data.qpos[qposadr + 1] = y
        mujoco.mj_forward(self.env.model, self.env.data)
        # 记录物体初始位置，供 run_task 规划 approach 轨迹
        self.object_init_pos = self._object_pos().copy()
        # place_target 的 z 取物体初始 z，保证落在桌面上（与物体同高）
        self.place_target[2] = self.object_init_pos[2]
        # 标记点：object（蓝）+ place_target（红）
        self.markers = [
            {"pos": self.object_init_pos.copy(), "rgba": [0.2, 0.6, 1, 1], "size": 0.025},
            {"pos": self.place_target.copy(),    "rgba": [1, 0, 0, 1],     "size": 0.025},
        ]
        return {"place_target": self.place_target.copy(),
                "object_pos": self.object_init_pos}

    def _object_pos(self) -> np.ndarray:
        """物体位置，来源由 env.pose_provider 决定（GT 或视觉）。"""
        return self.env.body_pose(self.object_name)[0]

    def compute_reward(self, obs: np.ndarray):
        obj_pos = self._object_pos()
        d_place = float(np.linalg.norm(obj_pos - self.place_target))
        ee_to_obj = float(np.linalg.norm(self.ee_pos - obj_pos))
        # 形状：靠近物体 + 抬高 + 到位
        reward = -ee_to_obj * 0.5 - d_place
        # 成功判定：物体到达目标区域 + 与桌面接触 + 夹爪已张开
        done = (d_place < self.success_thresh
                and self._on_table()
                and self.env.gripper.get_opening() > 0.5)
        info = {"ee_to_obj": ee_to_obj, "dist_to_place": d_place}
        if done:
            reward += 5.0
        return reward, done, info

    def _on_table(self) -> bool:
        """物体是否与桌面顶面接触。

        table_top 在 scene_table.xml 中是 geom 名（所属 body 名为 table），
        故用 mjOBJ_GEOM 查找其 geom id，再与 contact 的 geom1/geom2 比对。
        """
        oid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY,
                                self.object_name)
        table_top_gid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM,
                                          "table_top")
        obj_geoms = self.env.model.body_geomadr[oid]
        obj_geomn = self.env.model.body_geomnum[oid]
        obj_geom_ids = set(range(obj_geoms, obj_geoms + obj_geomn))
        for i in range(self.env.data.ncon):
            c = self.env.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if ((g1 in obj_geom_ids and g2 == table_top_gid) or
                    (g2 in obj_geom_ids and g1 == table_top_gid)):
                return True
        return False
