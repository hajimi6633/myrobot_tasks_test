"""Node 基类：统一节点接口（参数 / 通信 / 定时 / 生命周期 / 日志）。

模仿 ROS2 Node + LifecycleNode 的概念，但不强制状态机：
  - 主循环节点（threaded=False）：每控制拍由 Executor 调 on_tick
  - 旁路节点（threaded=True）：自带 worker 线程（渲染/视觉/安全），
    不进主循环，数据经 topic 的 latest 缓存交接
"""
from __future__ import annotations
from rclike.core import SimClock, Logger, ParameterStore
from rclike.comm import Bus, Publisher, Service


class Node:
    threaded = False        # 子类设 True 表示旁路线程节点

    def __init__(self, name: str, bus: Bus, clock: SimClock):
        self.name = name
        self.bus = bus
        self.clock = clock
        self.params = ParameterStore()
        self.log = Logger(name)
        self._timers: list[dict] = []        # {period, cb, last}

    # ---------- 参数 ----------
    def declare_parameter(self, name: str, default, desc: str = ""):
        return self.params.declare(name, default, desc)

    def get_parameter(self, name):
        return self.params.get(name)

    # ---------- 通信 ----------
    def create_publisher(self, topic: str) -> Publisher:
        return Publisher(self.bus, topic)

    def create_subscription(self, topic: str, cb):
        self.bus.topic(topic).subscribe(cb)

    def create_service(self, name: str, handler) -> Service:
        return self.bus.service(name, handler)

    def call_service(self, name: str, request):
        return self.bus.service(name).call(request)

    def latest(self, topic: str) -> tuple | None:
        """读话题最新值 (msg, stamp)。非阻塞，主循环消费旁路数据的标准方式。"""
        return self.bus.topic(topic).latest

    # ---------- 定时（主循环按频率触发，替代各脚本自己的频率循环）----------
    def create_timer(self, period: float, cb):
        self._timers.append({"period": period, "cb": cb, "last": None})

    # ---------- 生命周期（可选覆写，如 GraspNode 的 weld 状态）----------
    def on_configure(self): pass
    def on_activate(self): pass
    def on_deactivate(self): pass
    def on_cleanup(self): pass

    # ---------- 执行钩子 ----------
    def on_tick(self):
        """每个控制拍被 Executor 调用（仅主循环节点）。默认空实现。"""
        pass

    def start(self):
        """threaded 节点在此启动 worker 线程。"""
        pass

    def shutdown(self):
        pass
