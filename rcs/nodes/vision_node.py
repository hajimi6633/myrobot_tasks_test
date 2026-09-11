"""VisionNode：通用视觉 worker（threaded），按角色参数化。

双相机分工（对话第 5 轮）：
  e2h —— 眼在手外：粗定位 / 末端位姿估计 / 障碍物（跳帧 · 最新值语义）
  eih —— 眼在手上：目标精定位（相机位姿由 FK 给出，只解相机系下目标位姿）

检测算法经 _detect 钩子注入，实施顺序（对话第 5 轮决策）：
  1. 仿真验证期：ArUco + 真值外参打通全链路（验证架构而非算法）
  2. 算法替换期：YOLO 检测 + FoundationPose / ICP（仿真 GT 做误差监督）
  3. 真机迁移期：手眼标定（AX=XB）+ 外参标定
"""
from __future__ import annotations
import threading
import time as pytime

from rclike import Node


class VisionNode(Node):
    threaded = True

    def __init__(self, bus, clock, image_topic: str, out_topic: str,
                 role: str = "e2h", fps: float = 10.0):
        super().__init__(f"vision_{role}", bus, clock)
        assert role in ("e2h", "eih"), "role 必须是 e2h / eih"
        self.role = role
        self._period = 1.0 / fps
        self._pub = self.create_publisher(out_topic)
        self._img = None                     # 最新图像槽（原子替换引用）
        self._stop_evt = threading.Event()
        self.create_subscription(image_topic, self._on_img)

    def _on_img(self, img, stamp):
        self._img = (img, stamp)

    def _detect(self, img) -> dict:
        """检测钩子（子类覆写或策略注入）。返回：
        e2h: {"target_coarse": (pos, rot), "ee_pose": (pos, rot),
              "obstacles": [包围盒列表]}
        eih: {"target_fine": (pos, rot)}    # 相机系下 6D 位姿
        """
        raise NotImplementedError(
            "TODO: 接入视觉算法（ArUco PnP / YOLO+6D / FoundationPose / ICP）")

    def start(self):
        threading.Thread(target=self._loop, daemon=True,
                         name=self.name).start()

    def _loop(self):
        """worker：纯计算（只碰 numpy 图像，不碰 MjData），永远处理最新帧。"""
        while not self._stop_evt.is_set():
            snap = self._img
            if snap is None:
                pytime.sleep(0.001)
                continue
            img, stamp = snap
            result = self._detect(img)
            self._pub.publish(result, stamp=stamp)
            self._img = None                 # 跳帧不积压：旧帧直接丢弃
            pytime.sleep(self._period)

    def shutdown(self):
        self._stop_evt.set()
