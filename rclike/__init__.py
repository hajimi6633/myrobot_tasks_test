"""rclike：模仿 ROS2 概念的进程内微内核（不依赖 rclpy / DDS）。

组成（对话六轮决策的代码载体）：
  core      仿真时钟 / 节点日志 / 参数声明（替代散落常量）
  comm      Topic（零拷贝 + 最新值缓存）+ Service（请求/响应）
  node      Node 基类（参数 / 通信 / 定时 / 生命周期）
  executor  单线程执行器（主循环，确定性顺序 = 可复现）
  action    ActionServer（goal / feedback / result / 挂起恢复）
  frames    TF-lite 静态变换树（替代手工偏移维护）
"""
from rclike.core import SimClock, Logger, ParameterStore
from rclike.comm import Bus, Topic, Service, Publisher
from rclike.node import Node
from rclike.executor import Executor
from rclike.action import ActionServer, GoalHandle, GoalState
from rclike.frames import FrameTree

__all__ = [
    "SimClock", "Logger", "ParameterStore",
    "Bus", "Topic", "Service", "Publisher",
    "Node", "Executor",
    "ActionServer", "GoalHandle", "GoalState",
    "FrameTree",
]
