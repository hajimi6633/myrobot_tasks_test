"""Action Server：长任务的 goal → 周期 feedback → result / cancel。

支持挂起（安全停止触发）与恢复——对应人侵入后任务从挂起点续跑，
而非从头开始（对话第 6 轮慢通道设计）。
"""
from __future__ import annotations
from enum import Enum


class GoalState(Enum):
    RUNNING = "running"
    SUSPENDED = "suspended"     # 安全挂起：冻结等待恢复（区别于取消）
    SUCCEEDED = "succeeded"
    ABORTED = "aborted"         # 执行失败（如抓取力不足）
    CANCELED = "canceled"


class GoalHandle:
    """一次 goal 的执行句柄：携带目标、状态与结果。"""

    def __init__(self, goal):
        self.goal = goal
        self.state = GoalState.RUNNING
        self.result: dict = {}


class ActionServer:
    """单 goal 服务器。

    execute_step(handle) 由宿主节点每控制拍调用一次：
      返回 None          —— 继续运行
      返回 GoalState 终止态 —— 结束并发布 result
    """

    def __init__(self, node, name: str, execute_step):
        self.node = node
        self.name = name
        self.execute_step = execute_step
        self.handle: GoalHandle | None = None
        self.feedback_pub = node.create_publisher(f"{name}/feedback")
        self.result_pub = node.create_publisher(f"{name}/result")

    def send_goal(self, goal) -> GoalHandle:
        if self.handle is not None and self.handle.state is GoalState.RUNNING:
            raise RuntimeError("已有 goal 在执行（单 goal 服务器）")
        self.handle = GoalHandle(goal)
        self.node.log.info(f"接受 goal: {goal}")
        return self.handle

    def publish_feedback(self, fb):
        """宿主节点在 execute_step 中调用，发布阶段进度。"""
        self.feedback_pub.publish(fb, stamp=self.node.clock.now)

    def suspend(self):
        """安全慢通道触发：任务冻结（ArmController 同步冻结指令）。"""
        if self.handle and self.handle.state is GoalState.RUNNING:
            self.handle.state = GoalState.SUSPENDED
            self.node.log.warn("goal 已挂起（安全停止）")

    def resume(self):
        if self.handle and self.handle.state is GoalState.SUSPENDED:
            self.handle.state = GoalState.RUNNING
            self.node.log.info("goal 恢复执行")

    def cancel(self):
        if self.handle and self.handle.state in (GoalState.RUNNING,
                                                GoalState.SUSPENDED):
            self.handle.state = GoalState.CANCELED

    def spin(self):
        """每控制拍调用一次：仅 RUNNING 状态推进（SUSPENDED 自动跳过）。"""
        h = self.handle
        if h is None or h.state is not GoalState.RUNNING:
            return
        new_state = self.execute_step(h)
        if new_state is not None:
            h.state = new_state
            h.result["state"] = new_state.value
            self.result_pub.publish(h.result)
            self.node.log.info(f"goal 结束: {new_state.value}")
