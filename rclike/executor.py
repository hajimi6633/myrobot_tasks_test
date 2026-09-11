"""单线程执行器：唯一的主循环。

  - 顺序确定性 = 仿真可复现（同 seed 同结果，无锁无竞态）
  - 节点执行顺序 = add 顺序 = 数据依赖顺序（任务 → 控制 → 物理）
  - 定时器统一调度：不同节点可按不同频率运行（控制 50Hz、任务状态机每拍）
  - threaded 节点不进主循环：add 时只启动其 worker 线程
"""
from __future__ import annotations
from rclike.core import SimClock
from rclike.node import Node


class Executor:
    def __init__(self, clock: SimClock, dt: float = 0.05):
        self.clock = clock
        self.dt = dt                  # 控制拍周期（= n_substeps × timestep）
        self._nodes: list[Node] = []
        self._running = False

    def add(self, node: Node) -> Node:
        node.on_configure()
        node.on_activate()
        if node.threaded:
            node.start()              # 旁路 worker 线程，不占主循环
        else:
            self._nodes.append(node)
        return node

    def spin_once(self):
        """一个控制拍：主循环节点 on_tick → 到期定时器 → 推进仿真时钟。"""
        for node in self._nodes:
            node.on_tick()
        for node in self._nodes:
            self._fire_timers(node)
        self.clock.advance(self.dt)

    def _fire_timers(self, node: Node):
        t = self.clock.now
        for tm in node._timers:
            if tm["last"] is None or (t - tm["last"]) >= tm["period"] - 1e-9:
                tm["last"] = t
                tm["cb"]()

    def spin(self, n_ticks: int | None = None):
        """阻塞运行。n_ticks=None 表示无限运行（Ctrl+C 结束）。"""
        self._running = True
        i = 0
        try:
            while self._running and (n_ticks is None or i < n_ticks):
                self.spin_once()
                i += 1
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _shutdown(self):
        for node in self._nodes:
            node.on_deactivate()
            node.on_cleanup()
            node.shutdown()
        self._running = False
