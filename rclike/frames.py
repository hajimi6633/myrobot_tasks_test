"""TF-lite：静态坐标变换注册与查询。

替代旧代码中手工维护的 ee_to_gun_offset / grasp_rot
（src/env/tasks/charging_phases.py 的 _gun_to_ee / _gun_rot_to_ee）。

语义（与旧公式严格对齐）：
  t        存 parent 局部系下的常量偏移（抓取时记录，不随姿态变化）
  R        存 child 相对 parent 的旋转 R_child_parent（如 R_gun_ee）
  to_parent(child, pos, parent_rot)：世界系位置回 parent 端
      p_parent = p_child + R_parent_world @ t
      对应旧公式 ee_target = gun_target - ee_mat @ offset（t = -offset）
  rot_to_parent(child, rot)：世界系旋转回 parent 端
      R_parent = R_child_parent.T @ R_child
      对应旧公式 R_ee = grasp_rot.T @ R_gun_target
"""
from __future__ import annotations
import numpy as np


class FrameTree:
    def __init__(self):
        # child -> (parent, t[3], R[3,3])：t/R 均为 child 相对 parent 的量
        self._static: dict[str, tuple] = {}

    def set_static(self, parent: str, child: str, pos, rot):
        """注册静态变换（抓取成功时 GraspNode / 任务调用）。"""
        self._static[child] = (parent, np.asarray(pos, float).copy(),
                               np.asarray(rot, float).copy())

    def clear(self, child: str):
        """删除变换（枪体归还插座后）。"""
        self._static.pop(child, None)

    def has(self, child: str) -> bool:
        return child in self._static

    def to_parent(self, child: str, pos, parent_rot=None) -> np.ndarray:
        """child 世界目标 → parent 世界目标。
        parent_rot：parent 当前世界旋转（旧代码每步读当前 ee_mat）。"""
        _, t, _ = self._static[child]
        R = np.eye(3) if parent_rot is None else np.asarray(parent_rot, float)
        return np.asarray(pos, float) + R @ t

    def rot_to_parent(self, child: str, rot) -> np.ndarray:
        """child 世界旋转目标 → parent 世界旋转目标。"""
        _, _, R = self._static[child]
        return R.T @ np.asarray(rot, float)
