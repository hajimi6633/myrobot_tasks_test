"""充电枪 6 阶段任务：显式 Phase 状态机（Action Server，主循环）。

迁移映射：src/env/tasks/charging_phases.py（闭包 st 状态）→ 本文件。
重构收益：
  - 闭包 st dict → PhaseData 显式字段：可挂起 / 恢复 / 单元测试
  - 散落常量 → declare_parameter（launch YAML 可覆盖）
  - gun↔ee 偏移换算 → FrameTree（rclike.frames）
阶段流：抓取 → 移动 → 插枪(导纳) → 拔枪 → 归位 → 复位。
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import mujoco

from rclike import Node, ActionServer, GoalState, FrameTree
from src.control.trajectory import min_jerk, Trajectory
from src.control.impedance import AdmittanceController

# ---- 常量（迁移自 ChargingGunTask 类属性）----
GUN_BODY, SOCKET_BODY, CAR_SOCKET_BODY = "charging_gun_1", "charing_socket", "car_socket"
GRIPPER_BASE = "carry_shell"
GUN_SITE, GUN_SITE_1, GUN_SITE_2 = "gun_site", "gun_site_1", "gun_site_2"
CHARGING_SITE_1, CHARGING_SITE_2 = "charing_site_1", "charing_site_2"
CAR_SITE_1, CAR_SITE_2, CAR_SITE_DONE = "car_site_1", "car_site_2", "car_site_done"
EQ_SOCKET, EQ_GUN_EE = "eq_socgun_1", "eq_gun_ee"
GUN_COL_6, GUN_PAN_BODY = "gun_col_6", "gun_pan"
CHARGING_PAN_BODY = "charing_pan"
GRIP_OPEN, GRIP_CLOSE = 1.0, 0.0


@dataclass
class PhaseData:
    """单阶段：规划参数 + 运行时状态（原闭包 st 的显式化）。"""
    name: str
    kind: str = "hold"            # move|move_pos|hold|home|adm_insert|adm_home|insert_down
    grip: float = GRIP_OPEN
    # ---- 规划参数 ----
    site: str | None = None       # 目标 site 名
    rot_site: str | None = None   # 姿态对齐 site
    full: bool = False            # True=全姿态，False=仅 z 轴对齐
    back_z: float = 0.0          # 预接近：沿 site z 轴后退 (m)
    via: str | None = None        # move_pos 的 via 点键
    T: float = 2.0                # 运动时长 (s)
    hold_s: float = 1.0           # hold 时长 (s)
    n_steps: int = 0             # 导纳 / 下插段步数上限
    offset_gun: bool = False      # 抓取后对 gun_site 求 IK（利用臂展）
    exit_action: str = ""         # 退出动作：grasp | return_gun
    # ---- 运行时状态 ----
    entered: bool = False
    step_i: int = 0
    trajectory: object | None = None
    q_hold: np.ndarray | None = None
    nominal: np.ndarray | None = None       # 导纳名义目标
    prev_q_des: np.ndarray | None = None    # 链式 IK 初值（连续解连续）
    prev_actual: np.ndarray | None = None   # 目标变化限幅基准
    prev_dist: float = 1e9
    blocked_n: int = 0
    phase_done: bool = False
    phase_msg: str = ""


class ChargingAction(Node):
    """6 阶段充电枪任务（迁移自 charging_phases.build_all_phases）。"""

    SETTLE_S = 1.0                # 轨迹末尾 settle 保持段时长

    def __init__(self, bus, clock, sim, ik, grasp, pose):
        super().__init__("charging_action", bus, clock)
        self.sim, self.ik, self.grasp, self.pose = sim, ik, grasp, pose
        # ---- 参数化：原 charging_phases 顶部全部常量 ----
        self.declare_parameter("ctrl_dt", 0.05, "控制拍周期 (s)")
        self.declare_parameter("f_block", 40.0, "插枪推进力预算 (N)")
        self.declare_parameter("f_block_slow", 2.0, "归位段阻力阈值 (N)")
        self.declare_parameter("insert_step", 0.002, "名义推进步长 (m)")
        self.declare_parameter("insert_step_slow", 0.0006, "归位段慢速步长 (m)")
        self.declare_parameter("align_tol", 0.0015, "对心阈值 (m)")
        self.declare_parameter("align_step", 0.001, "对心闭环步长 (m)")
        self.declare_parameter("done_dist", 0.006, "到底距离阈值 (m)")
        self.declare_parameter("progress_win", 0.0005, "停滞判定窗口推进量 (m)")
        self.declare_parameter("max_travel", 0.005, "闭环段 IK 行程硬限 (rad)")
        self.declare_parameter("grasp_force_min", 0.5, "抓取成功最小力 (N)")
        self.declare_parameter("admittance_stiffness", 1000.0, "导纳刚度 K")

        self.dt_ctrl = self.get_parameter("ctrl_dt")
        self.gun_site_id = mujoco.mj_name2id(
            sim.model, mujoco.mjtObj.mjOBJ_SITE, GUN_SITE)
        self.frames = FrameTree()
        self.admittance = AdmittanceController(
            mass=1.0, stiffness=self.get_parameter("admittance_stiffness"),
            damping_ratio=1.0, max_delta=0.05)
        # via 点（进入阶段时求值，动态取两端点中点——固定常量曾撞臂体）
        self._vias = {
            "mid_cs2_car2": lambda: (self._sp(CHARGING_SITE_2) + self._sp(CAR_SITE_2)) / 2,
            "mid_car2_cs2": lambda: (self._sp(CAR_SITE_2) + self._sp(CHARGING_SITE_2)) / 2,
        }
        self.phases: list[PhaseData] = []
        self._pi = 0
        self._pub_qdes = self.create_publisher("/q_des")
        self.action = ActionServer(self, "charging", self._step)
        # 安全慢通道：zone 变化 → 挂起 / 恢复（快通道由 ArmController 冻结）
        self.create_subscription("/safety_state", self._on_safety)

    # ---------- 阶段序列（16 个子段，对应旧 build_phase1~6） ----------
    def _build_phases(self):
        P, O, C = PhaseData, GRIP_OPEN, GRIP_CLOSE
        self.phases = [
            # 1 抓取：预接近转正 → 到位 → 闭合 + weld 绑定
            P("1a0_pre", "move", site=GUN_SITE_2, T=3.0, rot_site=GUN_SITE_2,
              full=True, back_z=0.15, grip=O),
            P("1a", "move", site=GUN_SITE_2, T=1.5, rot_site=GUN_SITE_2,
              full=True, grip=O),
            P("1b", "hold", hold_s=1.0, grip=O),
            P("1c", "move", site=GUN_SITE_1, T=2.5, rot_site=GUN_SITE_1,
              full=True, grip=O),
            P("1d", "hold", hold_s=1.0, grip=O),
            P("1e", "hold", hold_s=1.0, grip=C, exit_action="grasp"),
            # 2 移动：枪 → 充电座口外 → via → 车插座口外
            P("2a", "move", site=CHARGING_SITE_2, T=2.0, offset_gun=True, grip=C),
            P("2b", "hold", hold_s=1.0, grip=C),
            P("2c1", "move_pos", via="mid_cs2_car2", T=2.5, offset_gun=True, grip=C),
            P("2c2", "move", site=CAR_SITE_2, T=2.5, offset_gun=True,
              rot_site=CAR_SITE_2, grip=C),
            P("2d", "hold", hold_s=1.0, grip=C),
            # 3 插枪（导纳 + 到底 / 停滞检测）
            P("3_insert", "adm_insert", n_steps=800, grip=C),
            # 4 拔枪：原路返回充电座口外
            P("4a", "move", site=CAR_SITE_2, T=1.5, offset_gun=True, grip=C),
            P("4b", "hold", hold_s=1.0, grip=C),
            P("4c1", "move_pos", via="mid_car2_cs2", T=2.5, offset_gun=True, grip=C),
            P("4c2", "move", site=CHARGING_SITE_2, T=2.5, offset_gun=True,
              rot_site=CHARGING_SITE_2, grip=C),
            # 5 归位：导纳回充电座 → 沿 z 下插到检测盘 → 归还插座 weld
            P("5a", "adm_home", n_steps=400, grip=C),
            P("5b", "insert_down", n_steps=300, grip=C, exit_action="return_gun"),
            # 6 复位：松爪 → 撤到枪尾外 → 回 home
            P("6a", "hold", hold_s=1.0, grip=O),
            P("6b", "move", site=GUN_SITE_2, T=2.0, grip=O),
            P("6c", "home", T=2.0, grip=O),
        ]
        self._pi = 0
        # 初始：枪固定在插座（迁移自 run 脚本的初始约束激活）
        self.grasp.constraints.set_active(EQ_SOCKET, True)
        mujoco.mj_forward(self.sim.model, self.sim.data)

    def send_goal(self, goal=None):
        self._build_phases()
        return self.action.send_goal(goal or {})

    # ---------- 主推进（每控制拍） ----------
    def on_tick(self):
        self.action.spin()

    def _step(self, handle):
        if self._pi >= len(self.phases):
            return GoalState.SUCCEEDED
        ph = self.phases[self._pi]
        if not ph.entered:
            self._enter_phase(ph)
            ph.entered = True
        q_des = self._advance(ph)
        if q_des is not None:
            self._pub_qdes.publish((q_des, ph.grip), stamp=self.clock.now)
            ph.step_i += 1
        done = self._check_done(ph)
        if done or ph.phase_done or ph.step_i >= self._phase_len(ph):
            self._exit_phase(ph)
            if ph.phase_done and "失败" in ph.phase_msg:
                handle.result["msg"] = ph.phase_msg
                return GoalState.ABORTED
            self.log.info(f"[{ph.name}] 完成 ({ph.step_i} 步) {ph.phase_msg}")
            self._pi += 1
            if self._pi >= len(self.phases):
                return GoalState.SUCCEEDED
        self.action.publish_feedback({"phase": ph.name, "step": ph.step_i})
        return None

    # ---------- 进入 / 推进 / 完成 / 退出（按 kind 分发） ----------
    def _enter_phase(self, ph: PhaseData):
        if ph.kind == "move":
            self._enter_move(ph)
        elif ph.kind == "move_pos":
            self._enter_move_pos(ph)
        elif ph.kind == "hold":
            self._enter_hold(ph)
        elif ph.kind == "home":
            self._enter_home(ph)
        elif ph.kind in ("adm_insert", "adm_home"):
            self._enter_admittance(ph)
        elif ph.kind == "insert_down":
            self._enter_insert_down(ph)

    def _advance(self, ph: PhaseData):
        if ph.kind in ("move", "move_pos", "home"):
            return ph.trajectory.at(ph.step_i)
        if ph.kind == "hold":
            return ph.q_hold
        if ph.kind == "adm_insert":
            return self._step_adm_insert(ph)
        if ph.kind == "adm_home":
            return self._step_adm_home(ph)
        if ph.kind == "insert_down":
            return self._step_insert_down(ph)

    def _check_done(self, ph: PhaseData) -> bool:
        if ph.kind == "adm_insert":
            return self._done_adm_insert(ph)
        if ph.kind == "adm_home":
            return self._done_adm_home(ph)
        if ph.kind == "insert_down":
            return self._done_insert_down(ph)
        return False                        # 轨迹 / hold 段按步数结束

    def _phase_len(self, ph: PhaseData) -> int:
        if ph.trajectory is not None:
            return len(ph.trajectory)
        if ph.kind == "hold":
            return int(ph.hold_s / self.dt_ctrl)
        return ph.n_steps

    def _exit_phase(self, ph: PhaseData):
        if ph.exit_action == "grasp":
            self._exit_grasp(ph)
        elif ph.exit_action == "return_gun":
            self._exit_return_gun()

    # ---------- 轨迹段实现（迁移 _move_phase / _plan） ----------
    def _enter_move(self, ph: PhaseData):
        sim = self.sim
        mujoco.mj_forward(sim.model, sim.data)
        q0 = sim.data.qpos[sim.arm_qposadr].copy()
        pos, site_m = sim.site_pose(ph.site)
        if ph.back_z > 0:                   # 预接近点：沿 site z 后退
            pos = pos - ph.back_z * site_m[:, 2]
        sid = self.gun_site_id if ph.offset_gun else None
        target_rot, z_align = None, False
        if ph.rot_site is not None:
            _, site_mat = sim.site_pose(ph.rot_site)
            # 抓取前 TF 未注册 → 单位变换（对应旧 ctx.grasp_rot 默认 eye）
            target_rot = site_mat if (ph.offset_gun or
                                      not self.frames.has("gun")) else \
                self.frames.rot_to_parent("gun", site_mat)
            z_align = not ph.full
        traj, q_goal = self._plan(pos, target_rot, ph.T, q_init=q0,
                                  site_id=sid, z_align_only=z_align)
        n_settle = max(1, int(self.SETTLE_S / self.dt_ctrl))
        ph.trajectory = Trajectory(
            np.vstack([traj.wp, np.tile(q_goal, (n_settle, 1))]), self.dt_ctrl)
        self.log.info(f"[{ph.name}] 规划 max|Δq|="
                     f"{np.max(np.abs(q_goal - q0)):.3f}rad")

    def _enter_move_pos(self, ph: PhaseData):
        target = self._vias[ph.via]()
        sim = self.sim
        mujoco.mj_forward(sim.model, sim.data)
        q0 = sim.data.qpos[sim.arm_qposadr].copy()
        sid = self.gun_site_id if ph.offset_gun else None
        traj, q_goal = self._plan(target, None, ph.T, q_init=q0, site_id=sid)
        n_settle = max(1, int(self.SETTLE_S / self.dt_ctrl))
        ph.trajectory = Trajectory(
            np.vstack([traj.wp, np.tile(q_goal, (n_settle, 1))]), self.dt_ctrl)

    def _enter_hold(self, ph: PhaseData):
        ph.q_hold = self.sim.data.qpos[self.sim.arm_qposadr].copy()

    def _enter_home(self, ph: PhaseData):
        q0 = self.sim.data.qpos[self.sim.arm_qposadr].copy()
        ph.trajectory = Trajectory(
            min_jerk(q0, self.sim.home_qpos, ph.T, self.dt_ctrl), self.dt_ctrl)

    def _plan(self, target_pos, target_rot, T, q_init=None,
             site_id=None, z_align_only=False):
        """IK 解目标 + min_jerk 轨迹（迁移 _plan）。"""
        q_cur = q_init if q_init is not None else \
            self.sim.data.qpos[self.sim.arm_qposadr].copy()
        q_goal = self.ik.ik.solve(target_pos, target_rot=target_rot,
                                  q_init=q_cur, site_id=site_id,
                                  z_align_only=z_align_only)
        return Trajectory(min_jerk(q_cur, q_goal, T, self.dt_ctrl),
                          self.dt_ctrl), q_goal

    # ---------- 导纳段实现（迁移 build_phase3 / 5a / 5b） ----------
    def _enter_admittance(self, ph: PhaseData):
        sim = self.sim
        mujoco.mj_forward(sim.model, sim.data)
        self.admittance.reset()
        ph.nominal = sim.site_pose(GUN_SITE)[0].copy()
        ph.prev_dist, ph.blocked_n, ph.prev_q_des, ph.prev_actual = 1e9, 0, None, None
        if ph.kind == "adm_home":
            self._pre_insert_retreat(ph)    # 4c 终点可能已穿入插座：先物理退出

    def _pre_insert_retreat(self, ph: PhaseData):
        """5a 专用：gun_col_6 已穿入充电插座时物理后退 5cm 脱离接触
        （防第一步解算 penetration 产生峰值力，迁移 enter_a 中段）。"""
        sim = self.sim
        if not sim.geom_body_collides(GUN_COL_6, SOCKET_BODY):
            return
        _, cs1_mat = sim.site_pose(CHARGING_SITE_1)
        back_pos = ph.nominal - 0.05 * cs1_mat[:, 2]   # -z 才是退出方向
        q_cur = sim.data.qpos[sim.arm_qposadr].copy()
        q_back = self.ik.ik.solve(back_pos, target_rot=None, q_init=q_cur,
                                  site_id=self.gun_site_id)
        sim.data.qpos[sim.arm_qposadr] = q_back
        mujoco.mj_forward(sim.model, sim.data)
        self.grasp.coupler.update(0.0)      # dt=0：仅同步位姿，不注入速度
        mujoco.mj_forward(sim.model, sim.data)
        ph.nominal = sim.site_pose(GUN_SITE)[0].copy()
        self.log.info("5a：预后退 5cm 脱离插座接触")

    def _admittance_common(self, ph, f_contact, axis):
        """导纳积分 + 轴向投影 + 目标限幅（phase3/5a/5b 公共部分）。"""
        self.admittance.step(f_contact, self.dt_ctrl)
        # 偏移投影到插座轴向：弹片切向力防横向漂移（曾漂移 1.2cm 卡死）
        d_ = self.admittance.delta
        self.admittance.delta[:] = axis * float(d_ @ axis)
        v_ = self.admittance.delta_dot
        self.admittance.delta_dot[:] = axis * float(v_ @ axis)
        actual = ph.nominal + self.admittance.delta
        # 单步目标限幅（防 IK 输入跳变传导为 q_des 突变）
        if ph.prev_actual is not None:
            diff = actual - ph.prev_actual
            dnorm = np.linalg.norm(diff)
            if dnorm > 0.005:
                actual = ph.prev_actual + 0.005 * diff / dnorm
        ph.prev_actual = actual.copy()
        # q_init 链式用上一步 q_des：连续两步的解天然连续
        q_cur = self.sim.data.qpos[self.sim.arm_qposadr].copy()
        q_init = ph.prev_q_des if ph.prev_q_des is not None else q_cur
        return actual, q_init

    def _step_adm_insert(self, ph: PhaseData):
        sim = self.sim
        f_contact = sim.contact_force_between(GUN_BODY, CAR_SOCKET_BODY)
        f_mag = float(np.linalg.norm(f_contact))
        car1, car1_mat = sim.site_pose(CAR_SITE_1)
        car_done = sim.site_pose(CAR_SITE_DONE)[0]
        ax = car1_mat[:, 2]                # 插座轴
        # 对心优先：实际横向偏差超阈值先横修（弹片口间隙仅 2.5mm）
        gun_act, _ = sim.site_pose(GUN_SITE)
        r_ = gun_act - car1
        lat = r_ - float(r_ @ ax) * ax
        lat_norm = np.linalg.norm(lat)
        if lat_norm > self.p("align_tol"):
            ph.nominal -= self.p("align_step") * lat / lat_norm
            if f_mag > self.p("f_block") and f_mag > 1e-3:
                ph.nominal += 0.001 * f_contact / f_mag
        elif f_mag < self.p("f_block"):
            # 已对准且阻力可接受：力相关步长缩放沿轴推进
            scale = max(0.1, 1.0 - f_mag / self.p("f_block"))
            d = car_done - ph.nominal
            nd = np.linalg.norm(d)
            if nd > self.p("done_dist"):
                ph.nominal = ph.nominal + self.p("insert_step") * scale * d / nd
        elif f_mag > 1e-3:
            ph.nominal += 0.001 * f_contact / f_mag     # 受阻回退卸力
        actual, q_init = self._admittance_common(ph, f_contact, ax)
        # 闭环段：禁多起点重试 + 行程硬限（欠收敛由下一步闭环自纠）
        q_des = self.ik.ik.solve(actual, target_rot=car1_mat, q_init=q_init,
                                  site_id=self.gun_site_id, z_align_only=True,
                                  retry=False, max_travel=self.p("max_travel"))
        ph.prev_q_des = q_des.copy()
        return q_des

    def _done_adm_insert(self, ph: PhaseData) -> bool:
        sim = self.sim
        if sim.body_collides_with(GUN_PAN_BODY, CAR_SOCKET_BODY):
            ph.phase_msg = "插枪到底：gun_pan 接触 car_socket"
            return True
        gun_p = sim.site_pose(GUN_SITE)[0]
        dist = np.linalg.norm(gun_p - sim.site_pose(CAR_SITE_DONE)[0])
        if dist < self.p("done_dist"):
            ph.phase_msg = "插枪到底：到达 car_site_done"
            return True
        # 停滞检测（窗口式累计推进：慢速推进不再被误判）
        if sim.body_collides_with(GUN_BODY, CAR_SOCKET_BODY):
            if dist < ph.prev_dist - self.p("progress_win"):
                ph.blocked_n = 0
                ph.prev_dist = dist
            else:
                ph.blocked_n += 1
            if ph.blocked_n > 100:
                ph.phase_msg = "插枪停滞：无进展"
                return True
        return False

    def _step_adm_home(self, ph: PhaseData):
        sim = self.sim
        f_contact = sim.contact_force_between(GUN_BODY, SOCKET_BODY)
        f_mag = float(np.linalg.norm(f_contact))
        cs1, cs1_mat = sim.site_pose(CHARGING_SITE_1)
        ax_ = cs1_mat[:, 2]
        gun_act, _ = sim.site_pose(GUN_SITE)
        r_ = gun_act - cs1
        lat = r_ - float(r_ @ ax_) * ax_
        lat_norm = np.linalg.norm(lat)
        if lat_norm > self.p("align_tol"):
            ph.nominal -= self.p("align_step") * lat / lat_norm
            if f_mag > self.p("f_block_slow") and f_mag > 1e-3:
                ph.nominal += 0.002 * f_contact / f_mag
        elif f_mag < self.p("f_block_slow"):
            scale = max(0.1, 1.0 - f_mag / self.p("f_block_slow"))
            d = cs1 - ph.nominal
            nd = np.linalg.norm(d)
            if nd > self.p("done_dist"):
                ph.nominal = ph.nominal + self.p("insert_step_slow") * scale * d / nd
        elif f_mag > 1e-3:
            ph.nominal += 0.002 * f_contact / f_mag
        actual, q_init = self._admittance_common(ph, f_contact, ax_)
        q_des = self.ik.ik.solve(actual, target_rot=cs1_mat, q_init=q_init,
                                 site_id=self.gun_site_id, z_align_only=True,
                                 retry=False, max_travel=0.02)
        ph.prev_q_des = q_des.copy()
        return q_des

    def _done_adm_home(self, ph: PhaseData) -> bool:
        sim = self.sim
        f_mag = float(np.linalg.norm(
            sim.contact_force_between(GUN_BODY, SOCKET_BODY)))
        if f_mag > 80.0:                   # 力安全限位
            ph.phase_msg = f"力安全限位 ({f_mag:.1f}N)"
            return True
        gun_p = sim.site_pose(GUN_SITE)[0]
        cs1 = sim.site_pose(CHARGING_SITE_1)[0]
        dist = np.linalg.norm(gun_p - cs1)
        if dist < 0.05:                    # 5cm 内即可，weld 会固定枪位
            ph.phase_msg = f"接近 charing_site_1 ({dist:.4f}m)"
            return True
        in_contact = (sim.geom_body_collides(GUN_COL_6, SOCKET_BODY) or
                      sim.body_collides_with(GUN_BODY, SOCKET_BODY))
        if in_contact:
            if dist < ph.prev_dist - self.p("progress_win"):
                ph.blocked_n = 0
                ph.prev_dist = dist
            else:
                ph.blocked_n += 1
            if ph.blocked_n > 100:
                ph.phase_msg = f"归位停滞 (距 cs1={dist:.4f}m)"
                return True
        return False

    def _enter_insert_down(self, ph: PhaseData):
        sim = self.sim
        mujoco.mj_forward(sim.model, sim.data)
        ph.nominal = sim.site_pose(GUN_SITE)[0].copy()
        ph.prev_q_des, ph.prev_actual = None, None

    def _step_insert_down(self, ph: PhaseData):
        sim = self.sim
        cs1_pos, cs1_mat = sim.site_pose(CHARGING_SITE_1)
        ax_ = cs1_mat[:, 2]                # +z 为插入方向
        target = cs1_pos + 0.1 * ax_       # 沿 z 下插 0.1m 至检测盘
        f_contact = sim.contact_force_between(GUN_BODY, SOCKET_BODY)
        f_mag = float(np.linalg.norm(f_contact))
        gun_act, _ = sim.site_pose(GUN_SITE)
        r_ = gun_act - cs1_pos
        lat = r_ - float(r_ @ ax_) * ax_
        lat_norm = np.linalg.norm(lat)
        if lat_norm > self.p("align_tol"):
            ph.nominal -= self.p("align_step") * lat / lat_norm
        elif float((target - ph.nominal) @ ax_) > self.p("insert_step_slow"):
            ph.nominal += self.p("insert_step_slow") * ax_
        if f_mag > self.p("f_block_slow") and f_mag > 1e-3:
            ph.nominal += 0.001 * f_contact / f_mag
        actual, q_init = self._admittance_common(ph, f_contact, ax_)
        q_des = self.ik.ik.solve(actual, target_rot=cs1_mat, q_init=q_init,
                                 site_id=self.gun_site_id, z_align_only=True,
                                 retry=False, max_travel=0.02)
        ph.prev_q_des = q_des.copy()
        return q_des

    def _done_insert_down(self, ph: PhaseData) -> bool:
        sim = self.sim
        if sim.body_collides_with(GUN_BODY, CHARGING_PAN_BODY):
            ph.phase_msg = "插枪到位：接触 charing_pan"
            return True
        if float(np.linalg.norm(
                sim.contact_force_between(GUN_BODY, SOCKET_BODY))) > 80.0:
            ph.phase_msg = "5b 力安全限位"
            return True
        cs1_pos, cs1_mat = sim.site_pose(CHARGING_SITE_1)
        depth = float((sim.site_pose(GUN_SITE)[0] - cs1_pos) @ cs1_mat[:, 2])
        if depth > 0.115:                  # 超深仍未触盘 = 异常
            ph.phase_msg = f"5b 深度超限 ({depth * 1000:.0f}mm)"
            return True
        return False

    # ---------- 阶段退出动作（weld 生命周期） ----------
    def _exit_grasp(self, ph: PhaseData):
        """1e 退出：抓取检测 → 激活枪-末端 weld → 注册 TF → 绑定 IK coupler。"""
        sim = self.sim
        f = sim.gripper.get_contact_force()
        if f < self.p("grasp_force_min"):
            ph.phase_done = True
            ph.phase_msg = f"抓取失败 f={f:.2f}N < {self.p('grasp_force_min')}N"
            return
        self.grasp._attach({"release_eq": EQ_SOCKET, "weld_eq": EQ_GUN_EE,
                             "anchor": GRIPPER_BASE, "object": GUN_BODY})
        self.ik.ik.coupler = self.grasp.coupler   # IK 迭代虚拟同步被夹物体
        # 注册 TF：ee→gun（替代 ee_to_gun_offset / grasp_rot 手工维护）
        gun_p, gun_mat = sim.site_pose(GUN_SITE)
        ee_p = sim.data.site_xpos[sim.ee_site_id]
        ee_mat = sim.data.site_xmat[sim.ee_site_id].reshape(3, 3)
        offset_local = ee_mat.T @ (gun_p - ee_p)
        self.frames.set_static("ee", "gun", -offset_local, gun_mat @ ee_mat.T)
        self.log.info(f"抓取成功 f={f:.2f}N")

    def _exit_return_gun(self):
        """5b 退出：先开插座 weld 再关末端 weld（枪任何时刻被持有）。"""
        self.grasp._detach({"socket_eq": EQ_SOCKET, "socket": SOCKET_BODY,
                            "weld_eq": EQ_GUN_EE, "object": GUN_BODY})
        self.ik.ik.coupler = None
        self.frames.clear("gun")

    # ---------- 安全慢通道 ----------
    def _on_safety(self, msg, stamp):
        if msg["zone"] == "stop":
            self.action.suspend()
        elif self.action.handle is not None and \
                self.action.handle.state.name == "SUSPENDED":
            # TODO: 恢复前人工确认策略（当前自动恢复）
            self.action.resume()

    # ---------- 辅助 ----------
    def p(self, name):
        return self.get_parameter(name)

    def _sp(self, site_name) -> np.ndarray:
        return self.sim.site_pose(site_name)[0]
