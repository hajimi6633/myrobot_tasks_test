"""RenderNode：旁路离屏渲染 worker（threaded，每相机一个实例）。

快照模式（对话第 4 轮，解决 MjData 并发竞争）：
  - 独立 MjData + 独立 GL 上下文，与主线程零共享可变状态
  - 只消费 /state_snapshot（qpos + 时间戳），不碰主 MjData
  - 时间戳随快照 → image → pose 全链传递，新鲜度判断有依据
  - model 只读共享（MuJoCo 保证线程安全）
"""
from __future__ import annotations
import threading
import time as pytime

import mujoco

from rclike import Node


class RenderNode(Node):
    threaded = True

    def __init__(self, bus, clock, model, camera_name: str,
                 out_topic: str, fps: float = 30.0):
        super().__init__(f"render_{camera_name}", bus, clock)
        self.model = model                       # 只读共享
        self.rdata = mujoco.MjData(model)        # 渲染专用 data（隔离）
        self.renderer = mujoco.Renderer(model)    # 独立离屏 GL 上下文
        self.camera = camera_name
        self._period = 1.0 / fps
        self._pub = self.create_publisher(out_topic)
        self._snap = None                        # 最新快照槽（原子替换引用）
        self._stop_evt = threading.Event()
        self.create_subscription("/state_snapshot", self._on_snap)

    def _on_snap(self, qpos, stamp):
        """主线程回调：仅存引用（微秒级），立即返回。"""
        self._snap = (qpos, stamp)

    def start(self):
        threading.Thread(target=self._loop, daemon=True,
                         name=self.name).start()

    def _loop(self):
        """worker：渲染慢于快照时自动跳帧（永远渲染最新快照，不积压）。"""
        while not self._stop_evt.is_set():
            snap = self._snap
            if snap is None:
                pytime.sleep(0.001)
                continue
            qpos, stamp = snap
            # 写入自己的 data；派生量重算也在这里（主线程无感）
            self.rdata.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.rdata)
            self.renderer.update_scene(self.rdata, camera=self.camera)
            img = self.renderer.render()          # np.uint8 (H, W, 3)
            self._pub.publish(img, stamp=stamp)   # 时间戳随快照传递
            self._snap = None
            pytime.sleep(self._period)

    def shutdown(self):
        self._stop_evt.set()
        self.renderer.close()
