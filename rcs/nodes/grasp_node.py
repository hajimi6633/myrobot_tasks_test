"""GraspNode：抓取 / weld 生命周期管理（主循环）。

迁移映射：src/control/grasp.py（GraspCoupler）+
src/control/constraints.py（ConstraintManager）→ 本节点。
旧类只依赖 model/data，原样复用，仅在外面包服务接口；
attach / detach 对应生命周期切换（对话第 1 轮决策）。

weld 激活顺序的坑（旧代码注释的原样保留）：
  激活前必须把当前相对位姿写入 eq_data——直接用编译时位姿激活，
  两 body 相差 1m+，会产生巨力瞬态（曾见 3 万 N 冲击）。
"""
from __future__ import annotations
import mujoco

from rclike import Node
from src.control.grasp import GraspCoupler
from src.control.constraints import ConstraintManager


class GraspNode(Node):
    def __init__(self, bus, clock, sim,
                 object_body: str = "charging_gun_1",
                 anchor_body: str = "carry_shell"):
        super().__init__("grasp_node", bus, clock)
        self.sim = sim
        self.coupler = GraspCoupler(sim.model, sim.data, object_body, anchor_body)
        self.constraints = ConstraintManager(sim.model, sim.data)
        # 服务接口：任务状态机在阶段切换时调用
        self.create_service("grasp/attach", self._attach)
        self.create_service("grasp/detach", self._detach)

    def _attach(self, req: dict | None = None) -> bool:
        """抓取成功：记录偏移 + 写当前相对位姿后激活 weld（零瞬态）。
        req: {release_eq: 插座weld名, weld_eq: 枪-末端weld名,
              anchor: 锚点body, object: 枪body}"""
        req = req or {}
        if req.get("release_eq"):
            self.constraints.set_active(req["release_eq"], False)
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.coupler.attach()
        self.constraints.set_weld_relpose(req["weld_eq"], req["anchor"],
                                          req["object"])
        self.constraints.set_active(req["weld_eq"], True)
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.log.info("抓取耦合已激活（weld）")
        return True

    def _detach(self, req: dict | None = None) -> bool:
        """归位：先激活插座 weld 再解除末端 weld——枪任何时刻
        都至少被一个 weld 持有（不悬空掉落）。"""
        req = req or {}
        self.constraints.set_weld_relpose(req["socket_eq"], req["socket"],
                                          req["object"])
        self.constraints.set_active(req["socket_eq"], True)
        self.constraints.set_active(req["weld_eq"], False)
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.coupler.detach()
        self.log.info("已解除耦合，枪归还插座 weld")
        return True
