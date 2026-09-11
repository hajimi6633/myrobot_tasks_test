"""SimNode：物理仿真节点——全系统唯一持有主 MjData 的地方。

迁移映射（绞杀式，旧代码保留对照）：
  src/env/arm_env.py 的仿真职责      → 本节点
  ArmEnv 的 obs 构建 / task 注入     → PoseNode / ChargingAction
  ArmEnv 的 post_step_hooks          → 订阅 /joint_states 的回调（tick 语义）
关键设计：属性命名与旧 ArmEnv 保持一致（model / data / arm_jids /
arm_qposadr / arm_dofadr / ee_site_id），使旧 IKSolver(env)
可直接以 IKSolver(sim) 复用，零改动迁移。
"""
from __future__ import annotations
import numpy as np
import mujoco

from rclike import Node
from src.config import load_yaml, project_path          # 复用旧配置工具
from src.control.gripper import Gripper                 # 只依赖 model/data，直接复用


class SimNode(Node):
    def __init__(self, bus, clock, scene_xml: str):
        super().__init__("sim_node", bus, clock)
        # ---- 参数（原 config/default.yaml sim 段）----
        self.declare_parameter("timestep", 0.002, "物理步长 (s)")
        self.declare_parameter("n_substeps", 25, "每控制拍物理子步数")
        self.declare_parameter("gravity_comp", True, "重力补偿前馈")

        # ---- 模型与索引（迁移自 ArmEnv.__init__）----
        self.model = mujoco.MjModel.from_xml_path(project_path(scene_xml))
        self.model.opt.timestep = self.get_parameter("timestep")
        self.data = mujoco.MjData(self.model)

        ur5e = load_yaml("config/ur5e.yaml")
        self.arm_joint_names = ur5e["joints"]["names"]
        self.arm_jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                         for n in self.arm_joint_names]
        self.arm_qposadr = np.array([self.model.jnt_qposadr[j]
                                     for j in self.arm_jids])
        self.arm_dofadr = np.array([self.model.jnt_dofadr[j]
                                    for j in self.arm_jids])
        self.arm_dof = len(self.arm_joint_names)
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, ur5e["ee_site"])
        self.home_qpos = np.array(ur5e["joints"]["home_qpos"], dtype=float)

        # actuator 查表（迁移自 ArmEnv._collect_actuators）
        self.act_act_ids = self._collect_actuators("act_")
        gcfg = load_yaml("config/gripper.yaml").get("gripper", {})
        self.gripper = Gripper(self.model, self.data,
                               left_joint=gcfg.get("left_joint", "ll_grasp_joint"),
                               right_joint=gcfg.get("right_joint", "rl_grasp_joint"),
                               left_body=gcfg.get("left_body", "finger_left"),
                               right_body=gcfg.get("right_body", "finger_right"))

        # ---- 通信 ----
        self.pub_states = self.create_publisher("/joint_states")
        self.pub_snapshot = self.create_publisher("/state_snapshot")
        self._cmd = self.bus.topic("/joint_cmd")     # 读最新值（安全钳制后）

    def _collect_actuators(self, prefix: str) -> dict:
        """按 actuator name 前缀收集 {被驱动关节名: actuator_id}。"""
        out = {}
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name and name.startswith(prefix):
                jid = self.model.actuator_trnid[i, 0]
                jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                out[jname] = i
        return out

    def reset(self):
        """物理重置到 home（迁移自 ArmEnv.reset）。"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qposadr] = self.home_qpos
        self.data.qpos[self.gripper.qposadr_l] = self.gripper.range_l[0]
        self.data.qpos[self.gripper.qposadr_r] = self.gripper.range_r[1]
        mujoco.mj_forward(self.model, self.data)

    def on_tick(self):
        # 1) 消费最新指令（ArmController 钳制后的输出）
        cmd = self._cmd.latest
        if cmd is not None:
            self._apply_cmd(cmd[0])
        # 2) 物理推进（迁移自 ArmEnv.step 子步循环）
        for _ in range(self.get_parameter("n_substeps")):
            if self.get_parameter("gravity_comp"):
                mujoco.mj_forward(self.model, self.data)
                self.data.qfrc_applied[self.arm_dofadr] = \
                    self.data.qfrc_bias[self.arm_dofadr]
            mujoco.mj_step(self.model, self.data)
        # 3) 发布状态与快照（时间戳 = 仿真时钟，全链新鲜度基准）
        t = self.clock.now
        self.pub_states.publish(self.joint_state(), stamp=t)
        # 微秒级快照：qpos 深拷贝一次，之后渲染/视觉/安全全链传引用
        self.pub_snapshot.publish(self.data.qpos.copy(), stamp=t)

    def joint_state(self):
        return (self.data.qpos[self.arm_qposadr].copy(),
                self.data.qvel[self.arm_dofadr].copy())

    # ---------- 查询（迁移自旧 ArmEnv，任务层 / done 判定用） ----------
    def site_pose(self, name: str):
        """site 世界位姿 (pos[3], rot_mat[3,3])。"""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError(f"site '{name}' not found")
        return (self.data.site_xpos[sid].copy(),
                self.data.site_xmat[sid].reshape(3, 3).copy())

    def _subtree_geoms(self, bid: int) -> set:
        """body 及其全部后代的 geom id 集合（枪头圆盘在子 body，
        接触/力查询必须用子树集合，否则静默漏检）。"""
        bodies = {bid}
        # MuJoCo 中 parent id 恒小于 child id，单轮扩张即可
        for k in range(self.model.nbody):
            if self.model.body_parentid[k] in bodies:
                bodies.add(k)
        geoms: set = set()
        for b in bodies:
            geoms.update(range(self.model.body_geomadr[b],
                               self.model.body_geomadr[b] + self.model.body_geomnum[b]))
        return geoms

    def body_collides_with(self, body_name: str, other_body_name: str) -> bool:
        """两 body（含子树）当前是否存在接触。"""
        b1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        b2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, other_body_name)
        if b1 < 0 or b2 < 0:
            return False
        g1s, g2s = self._subtree_geoms(b1), self._subtree_geoms(b2)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if ((c.geom1 in g1s and c.geom2 in g2s) or
                    (c.geom2 in g1s and c.geom1 in g2s)):
                return True
        return False

    def geom_body_collides(self, geom_name: str, body_name: str) -> bool:
        """指定 geom 是否与某 body（含子树）接触。"""
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if gid < 0 or bid < 0:
            return False
        gbs = self._subtree_geoms(bid)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if ((c.geom1 == gid and c.geom2 in gbs) or
                    (c.geom2 == gid and c.geom1 in gbs)):
                return True
        return False

    def contact_force_between(self, body1_name: str, body2_name: str):
        """body1 受到的来自 body2 的接触合力（世界系 [fx,fy,fz]）。"""
        b1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
        b2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_name)
        if b1 < 0 or b2 < 0:
            return np.zeros(3)
        g1s, g2s = self._subtree_geoms(b1), self._subtree_geoms(b2)
        f_total = np.zeros(3)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            f = np.zeros(6)
            R = c.frame.reshape(3, 3)
            if c.geom1 in g1s and c.geom2 in g2s:
                mujoco.mj_contactForce(self.model, self.data, i, f)
                f_total -= R @ f[:3]       # 法向 b1→b2，body1 受力取负
            elif c.geom2 in g1s and c.geom1 in g2s:
                mujoco.mj_contactForce(self.model, self.data, i, f)
                f_total += R @ f[:3]
        return f_total

    def _apply_cmd(self, cmd):
        """写 ctrl（迁移自 ArmEnv._apply_action）。cmd = (q_arm[6], grip_ratio)。"""
        q_arm, grip = cmd
        self.data.ctrl[:] = 0.0
        for i, name in enumerate(self.arm_joint_names):
            self.data.ctrl[self.act_act_ids[name]] = q_arm[i]
        t_l, _ = self.gripper.set_target(float(grip))
        if self.gripper.left_joint in self.act_act_ids:
            self.data.ctrl[self.act_act_ids[self.gripper.left_joint]] = t_l
