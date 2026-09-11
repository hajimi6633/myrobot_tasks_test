"""launch：节点栈组装器（等价 ROS2 launch + 参数注入）。

组装原则：
  - 主循环节点 add 顺序 = 执行顺序 = 数据依赖顺序
    （任务 → 控制闸门 → 物理；任务每拍经 /q_des 通知控制层，
    一拍延迟与真实 ROS 控制器一致）
  - threaded 节点（渲染 / 视觉 / 安全）只启动 worker，不进主循环
  - 参数覆盖：节点声明默认值后，可在此用 params.override 注入 YAML 值
"""
from __future__ import annotations

from rclike import SimClock, Bus, Executor

from rcs.nodes.sim_node import SimNode
from rcs.nodes.arm_controller import ArmController
from rcs.nodes.ik_service import IKService
from rcs.nodes.grasp_node import GraspNode
from rcs.nodes.pose_node import PoseNode
from rcs.nodes.render_node import RenderNode
from rcs.nodes.vision_node import VisionNode
from rcs.nodes.safety_node import SafetyNode
from rcs.tasks.charging_action import ChargingAction


def build_charging_stack(scene_xml: str, vision: bool = False):
    """组装充电枪任务全栈。返回 (executor, 句柄 dict)。"""
    clock, bus = SimClock(), Bus()

    # ---- 主循环节点 ----
    sim = SimNode(bus, clock, scene_xml)
    ctrl = ArmController(bus, clock)
    ik = IKService(bus, clock, sim)            # 旧 IKSolver 换绑 sim
    grasp = GraspNode(bus, clock, sim)         # 旧 Coupler/Constraints 包装
    pose = PoseNode(bus, clock, sim)
    task = ChargingAction(bus, clock, ik, grasp, pose)

    # ---- 旁路线程节点（每相机一渲染 + 安全哨兵）----
    render_e2h = RenderNode(bus, clock, sim.model, "track", "/image_e2h")
    render_eih = RenderNode(bus, clock, sim.model, "wrist", "/image_eih")
    safety = SafetyNode(bus, clock, "/image_e2h")
    vision_e2h = VisionNode(bus, clock, "/image_e2h",
                            "/ee_pose_vision", "e2h") if vision else None
    vision_eih = VisionNode(bus, clock, "/image_eih",
                            "/target_fine", "eih") if vision else None

    ex = Executor(clock, dt=0.05)             # 50Hz 控制拍
    for n in (task, ctrl, ik, grasp, pose, sim):      # 主循环顺序
        ex.add(n)
    for n in (render_e2h, render_eih, safety,
              vision_e2h, vision_eih):
        if n is not None:
            ex.add(n)                                 # threaded：只启动 worker

    handles = {"task": task, "ctrl": ctrl, "sim": sim, "ik": ik,
               "grasp": grasp, "pose": pose, "safety": safety,
               "executor": ex}
    return ex, handles
