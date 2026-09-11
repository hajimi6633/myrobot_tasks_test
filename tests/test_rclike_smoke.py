"""rclike 内核冒烟测试：不依赖 MuJoCo，验证六轮对话的核心机制。

覆盖：零拷贝通信 / 最新值缓存 / 执行器顺序与时钟 / 定时器频率 /
服务调用 / Action 挂起恢复 / TF-lite 静态变换。
运行：python -m pytest tests/test_rclike_smoke.py -q
"""
from __future__ import annotations
import numpy as np

from rclike import (SimClock, Bus, Node, Executor, ActionServer,
                    GoalState, FrameTree)


def test_topic_zero_copy_and_latest():
    """零拷贝：订阅者收到的是同一对象；latest 缓存最新值。"""
    bus = Bus()
    t = bus.topic("/t")
    got = []
    t.subscribe(lambda msg, stamp: got.append((msg, stamp)))
    arr = np.ones(3)
    t.publish(arr, stamp=1.0)
    assert got[0][0] is arr                 # 同一引用，无拷贝
    assert t.latest == (arr, 1.0)


def test_executor_order_and_clock():
    """执行顺序 = add 顺序（确定性）；时钟按拍推进。"""
    clock, bus = SimClock(), Bus()
    order = []

    class A(Node):
        def on_tick(self): order.append("a")

    class B(Node):
        def on_tick(self): order.append("b")

    ex = Executor(clock, dt=0.05)
    ex.add(A("a", bus, clock))
    ex.add(B("b", bus, clock))
    ex.spin(n_ticks=2)
    assert order == ["a", "b", "a", "b"]
    assert abs(clock.now - 0.1) < 1e-9


def test_timer_frequency():
    """定时器按周期触发（0.1s 周期 / 0.05s 拍 = 每 2 拍一次）。"""
    clock, bus = SimClock(), Bus()
    n = Node("n", bus, clock)
    calls = []
    n.create_timer(0.1, lambda: calls.append(clock.now))
    ex = Executor(clock, dt=0.05)
    ex.add(n)
    ex.spin(n_ticks=5)                      # 0.25s 内应触发 3 次
    assert len(calls) == 3


def test_service_call():
    """服务 = 进程内直接函数调用。"""
    bus, clock = Bus(), SimClock()
    node = Node("n", bus, clock)
    node.create_service("add", lambda req: req["a"] + req["b"])
    assert bus.service("add").call({"a": 1, "b": 2}) == 3


def test_action_suspend_resume():
    """Action：挂起期间不推进；恢复后从挂起点续跑（安全慢通道语义）。"""
    clock, bus = SimClock(), Bus()

    class Task(Node):
        def __init__(self):
            super().__init__("task", bus, clock)
            self.steps = 0
            self.action = ActionServer(self, "t", self._step)

        def on_tick(self):                 # 与 ChargingAction 相同：每拍驱动
            self.action.spin()

        def _step(self, handle):
            self.steps += 1
            return None                      # 永不自然结束

    t = Task()
    ex = Executor(clock, dt=0.05)
    ex.add(t)
    t.action.send_goal({})
    ex.spin(n_ticks=2)
    assert t.steps == 2
    t.action.suspend()                       # 模拟人侵入
    ex.spin(n_ticks=3)
    assert t.steps == 2                      # 挂起：零推进
    assert t.action.handle.state is GoalState.SUSPENDED
    t.action.resume()                        # 模拟人离开 + 确认
    ex.spin(n_ticks=1)
    assert t.steps == 3                      # 从挂起点续跑


def test_frames_static_transform():
    """TF-lite：局部偏移随 parent 姿态旋转；旋转回退用 R.T。"""
    ft = FrameTree()
    # 旧语义：ee_target = gun_target - ee_mat @ offset → t = -offset, R = R_gun_ee
    offset = np.array([0.1, 0.0, 0.0])
    R_gun_ee = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 绕 z 转 90°
    ft.set_static("ee", "gun", -offset, R_gun_ee)

    gun_target = np.array([1.0, 0.5, 0.3])
    ee_mat = np.eye(3)
    expect = gun_target - ee_mat @ offset            # 旧 _gun_to_ee 公式
    assert np.allclose(ft.to_parent("gun", gun_target, ee_mat), expect)

    # 旋转回退：R_ee = R_gun_ee.T @ R_gun（旧 _gun_rot_to_ee 公式）
    R_gun = np.eye(3)
    assert np.allclose(ft.rot_to_parent("gun", R_gun), R_gun_ee.T @ R_gun)

    # 清除后不可再查（枪已归还插座）
    ft.clear("gun")
    try:
        ft.to_parent("gun", gun_target)
        assert False, "应抛 KeyError"
    except KeyError:
        pass
