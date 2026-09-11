"""TF-lite：静态坐标变换注册与查询。

替代旧代码中手工维护的 ee_to_gun_offset / grasp_rot
（src/env/tasks/charging_phases.py 的 _gun_to_ee / _gun_rot_to_ee）。

约定：set_static(parent, child, t, R) 存储 child 相对 parent 的
平移 t 与旋转 R；查询时把 child 系下的量变换回 parent 系。
抓取成功时注册 ee_link→gun_site，归还后 clear。
"""
from __future__ import annotations
import numpy as np


class FrameTree:
    def __init__(self):
        # child -> (parent, t[3], R[3,3])：child 相对 parent 的变换
        self._static: dict[str, tuple] = {}

    def set_static(self, parent: str, child: str, pos, rot):
        """注册静态变换（如抓取成功时 GraspNode 调用）。"""
        self._static[child] = (parent, np.asarray(pos, float).copy(),
                               np.asarray(rot, float).copy())

    def clear(self, child: str):
        """删除变换（如枪体归还插座后）。"""
        self._static.pop(child, None)

    def to_parent(self, child: str, pos) -> np.ndarray:
        """child 系位置 → parent 系位置（如 gun 目标 → ee 目标）。"""
        _, t, R = self._static[child]
        return np.asarray(pos, float) + R @ t

    def rot_to_parent(self, child: str, rot) -> np.ndarray:
        """child 系旋转 → parent 系旋转（枪姿态目标 → 末端姿态目标）。"""
        _, _, R = self._static[child]
        return R @ np.asarray(rot, float)
