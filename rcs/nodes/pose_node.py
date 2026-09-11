"""PoseNode：位姿聚合与新鲜度管理（主循环）。

数据源优先级（功能语义，注意与安全链路的 fail-safe 相反）：
  视觉新鲜（< freshness 窗口）→ 用视觉；过期 → 回退 GT（仿真期）
handoff（对话第 5 轮）：粗定位（E2H）⇄ 精定位（EIH）切换，
切换判据在任务状态机（目标进入 EIH 视野且粗对准达标）。
"""
from __future__ import annotations

from rclike import Node


class PoseNode(Node):
    def __init__(self, bus, clock, sim):
        super().__init__("pose_node", bus, clock)
        self.sim = sim
        self.declare_parameter("freshness_s", 0.15, "视觉数据新鲜度窗口 (s)")
        self._vis_ee = None        # (msg, stamp)：E2H 末端位姿估计
        self._vis_fine = None      # EIH 精定位结果
        self.mode = "coarse"       # handoff 状态：coarse → fine
        self.create_subscription("/ee_pose_vision", self._on_vis_ee)
        self.create_subscription("/target_fine", self._on_fine)
        # 查询服务：下游（任务/控制/RL）的统一位姿入口
        self.create_service("get_ee_pose", self._get_ee_pose)
        self.create_service("get_target", self._get_target)

    # ---- 视觉结果回调（只存引用，立即返回）----
    def _on_vis_ee(self, msg, stamp):
        self._vis_ee = (msg, stamp)

    def _on_fine(self, msg, stamp):
        self._vis_fine = (msg, stamp)

    def handoff(self, mode: str):
        """任务状态机调用：伺服段切 fine，粗接近段回 coarse。"""
        assert mode in ("coarse", "fine")
        self.mode = mode
        self.log.info(f"handoff -> {mode}")

    # ---- 查询：新鲜则视觉，过期回退 GT ----
    def _get_ee_pose(self, req=None):
        vis = self._vis_ee
        if vis and (self.clock.now - vis[1]) < self.get_parameter("freshness_s"):
            return vis[0]
        return self._gt_ee_pose()

    def _get_target(self, req=None):
        """目标位姿：fine 模式优先 EIH 精定位，否则 E2H 粗定位。"""
        if self.mode == "fine" and self._vis_fine is not None:
            return self._vis_fine[0]
        # TODO: E2H 粗定位结果转发（VisionNode e2h 的 target_coarse）
        return None

    def _gt_ee_pose(self):
        """GT 回退：直接读仿真派生量（主线程内调用，无竞争）。
        真机上无 GT，此处应改为报错或冻结上一次值。"""
        sid = self.sim.ee_site_id
        return (self.sim.data.site_xpos[sid].copy(),
                self.sim.data.site_xmat[sid].reshape(3, 3).copy())
