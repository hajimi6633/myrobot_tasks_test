"""进程内通信：Topic（零拷贝发布/订阅）+ Service（请求/响应）。

设计要点：
  - publish 同步调用订阅回调，msg 传引用（numpy 零拷贝，无序列化）
  - 每条消息开销 ≈ 字典查找 + 函数调用（纳秒级）
  - 附带 latest 最新值缓存：旁路线程数据的消费端标准读取方式
"""
from __future__ import annotations
from typing import Callable, Any


class Topic:
    """话题通道。一个话题 = 一个回调注册表 + 一份最新值缓存。"""

    def __init__(self, name: str):
        self.name = name
        self._subs: list[Callable] = []
        self._latest: tuple | None = None    # (msg, stamp)

    def publish(self, msg: Any, stamp: float | None = None):
        """发布：直接传引用调用全部订阅回调（确定性顺序）。"""
        self._latest = (msg, stamp)
        for cb in list(self._subs):           # 拷贝后再遍历，防回调中增删订阅
            cb(msg, stamp)

    def subscribe(self, cb: Callable) -> Callable:
        self._subs.append(cb)
        return cb

    @property
    def latest(self) -> tuple | None:
        """最近一次 (msg, stamp)。非阻塞读取旁路线程数据的标准入口。"""
        return self._latest


class Service:
    """服务：进程内直接函数调用（如 IK 求解、weld 切换这类请求-响应）。"""

    def __init__(self, name: str, handler: Callable | None):
        self.name = name
        self.handler = handler

    def call(self, request: Any) -> Any:
        if self.handler is None:
            raise RuntimeError(f"服务 '{self.name}' 尚未注册 handler")
        return self.handler(request)


class Bus:
    """话题/服务注册表：launch 时创建一个，注入所有节点。"""

    def __init__(self):
        self._topics: dict[str, Topic] = {}
        self._services: dict[str, Service] = {}

    def topic(self, name: str) -> Topic:
        if name not in self._topics:
            self._topics[name] = Topic(name)
        return self._topics[name]

    def service(self, name: str, handler: Callable | None = None) -> Service:
        if name not in self._services:
            self._services[name] = Service(name, handler)
        return self._services[name]


class Publisher:
    """发布句柄：包装 Topic.publish，接口语义对齐 ROS2。"""

    def __init__(self, bus: Bus, name: str):
        self.name = name
        self._topic = bus.topic(name)

    def publish(self, msg: Any, stamp: float | None = None):
        self._topic.publish(msg, stamp)
