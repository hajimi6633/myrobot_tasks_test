"""位姿提供者抽象层：解耦位姿数据来源。

设计目标：
  - 同一套上层代码（env/task/IK）既可消费仿真真值，也可消费相机视觉算法输出。
  - 通过 config/default.yaml 的 pose.source 切换：gt | vision
  - 后续接入相机视觉算法时，只需实现 VisionPoseProvider 的 _detect_* 钩子。

统一位姿表示： (pos: np.ndarray[3], rot: np.ndarray[3,3])
  pos  世界系位置
  rot  世界系→本体旋转矩阵（行优先 3x3，与 MuJoCo xmat 一致）
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import mujoco


class PoseProvider(ABC):
    """位姿提供者接口。每步 step 前由 env 调用 update() 同步内部状态。"""

    def attach(self, env):
        """env 构造时注入，便于访问 model/data/camera。"""
        self.env = env
        # 级联 attach 子 provider（Vision 的 fallback 也需要 env 引用）
        child = getattr(self, "fallback", None)
        if child is not None and getattr(child, "env", None) is None:
            child.attach(env)

    @abstractmethod
    def update(self):
        """每步物理推进后调用，刷新内部缓存（GT 直接读，Vision 跑检测）。"""

    @abstractmethod
    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """返回末端执行器 (pos[3], rot[3,3])。"""

    @abstractmethod
    def get_body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """返回指定 body 的 (pos[3], rot[3,3])。"""


class GroundTruthPoseProvider(PoseProvider):
    """直接从 MuJoCo data 读取真值。零延迟、无噪声。"""

    def update(self):
        pass  # 真值实时反映在 data 中，无需缓存

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        sid = self.env.ee_site_id
        pos = self.env.data.site_xpos[sid].copy()
        rot = self.env.data.site_xmat[sid].reshape(3, 3).copy()
        return pos, rot

    def get_body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        bid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"body '{name}' not found")
        pos = self.env.data.xpos[bid].copy()
        rot = self.env.data.xmat[bid].reshape(3, 3).copy()
        return pos, rot


class VisionPoseProvider(PoseProvider):
    """相机视觉算法位姿提供者（占位骨架）。

    接入步骤（后续实现）：
      1. 在 __init__ 中初始化相机参数、检测模型（如 ARUCO/YOLO+PnP）
      2. update() 中调用 mujoco.mj_render 渲染或读取真实相机图像
      3. _detect_ee / _detect_body 实现目标检测 + 位姿估计
      4. 估计结果缓存到 self._ee_cache / self._body_cache，供 get_* 读取

    可注入可选的真值 provider 作为 fallback（视觉丢失时回退）。
    """

    def __init__(self, fallback: PoseProvider | None = None,
                 camera_name: str = "track"):
        self.fallback = fallback
        self.camera_name = camera_name
        self._ee_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._body_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def update(self):
        """TODO: 渲染相机图像并运行视觉检测，结果写入 cache。

        示例骨架:
            img = self._render_camera()
            self._ee_cache = self._detect_ee(img)
            self._body_cache["object"] = self._detect_body("object", img)
        """
        # 当前未实现检测：若设了 fallback 则刷新 fallback，cache 留空走 fallback
        if self.fallback is not None:
            self.fallback.update()

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self._ee_cache is not None:
            return self._ee_cache
        if self.fallback is not None:
            return self.fallback.get_ee_pose()
        raise NotImplementedError(
            "VisionPoseProvider: 未实现 _detect_ee，且无 fallback。"
            "请在 update() 中实现视觉检测，或构造时传入 fallback=GroundTruthPoseProvider()。")

    def get_body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        if name in self._body_cache:
            return self._body_cache[name]
        if self.fallback is not None:
            return self.fallback.get_body_pose(name)
        raise NotImplementedError(
            f"VisionPoseProvider: 未实现 _detect_body('{name}')，且无 fallback。")


def make_pose_provider(source: str, env) -> PoseProvider:
    """工厂：根据 config pose.source 创建 provider。"""
    source = source.lower()
    if source == "gt":
        return GroundTruthPoseProvider()
    if source == "vision":
        # 默认以 GT 作为 fallback，避免未实现视觉时整条链路崩溃
        return VisionPoseProvider(fallback=GroundTruthPoseProvider())
    raise ValueError(f"unknown pose source: {source} (expect 'gt' | 'vision')")
