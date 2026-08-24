"""ArmEnv：封装 MuJoCo 仿真，统一 reset/step 接口（position 模式）。

action 拼接：[arm_action(6), gripper_open_ratio(1)]
  - arm_action = 6 个目标关节角 (rad)
  - gripper_open_ratio ∈ [0,1]：0 闭合，1 张开
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import mujoco

from src.config import load_yaml, project_path
from src.control.gripper import Gripper
from src.perception.pose_provider import PoseProvider, make_pose_provider


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: dict


class ArmEnv:
    ACTION_MODES = ("position",)

    def __init__(self, config_rel: str = "config/default.yaml",
                 action_mode: str = "position", task=None,
                 pose_source: str | None = None,
                 pose_provider: PoseProvider | None = None,
                 gravity_comp: bool = False):
        self.cfg = load_yaml(config_rel)
        self.ur5e_cfg = load_yaml("config/ur5e.yaml")
        self.gripper_cfg = load_yaml("config/gripper.yaml")

        if action_mode not in self.ACTION_MODES:
            raise ValueError(f"action_mode must be {self.ACTION_MODES}; "
                             f"torque 模式暂未启用")
        self.action_mode = action_mode
        # 重力补偿前馈：position 模式下用 qfrc_bias 补偿重力，消除稳态误差
        self.gravity_comp = gravity_comp

        # 加载模型（独立完整场景，支持相对项目根或绝对路径）
        xml_path = project_path(self.cfg["model"]["scene_xml"])
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = self.cfg["sim"]["timestep"]
        self.data = mujoco.MjData(self.model)

        # 关节索引（6 自由度臂）
        self.arm_joint_names = self.ur5e_cfg["joints"]["names"]
        self.arm_jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                         for n in self.arm_joint_names]
        self.arm_qposadr = np.array([self.model.jnt_qposadr[j] for j in self.arm_jids])
        self.arm_dofadr = np.array([self.model.jnt_dofadr[j] for j in self.arm_jids])
        self.arm_dof = len(self.arm_joint_names)

        # actuator 名称 -> id（act_act_ids 同时包含 arm 关节和 finger）
        self.act_act_ids = self._collect_actuators("act_")

        # 末端 site
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.ur5e_cfg["ee_site"])

        # 夹爪：joint/body 名从 config/gripper.yaml 读取，更换模型无需改代码
        gcfg = self.gripper_cfg.get("gripper", {})
        self.gripper = Gripper(
            self.model, self.data,
            left_joint=gcfg.get("left_joint", "ll_grasp_joint"),
            right_joint=gcfg.get("right_joint", "rl_grasp_joint"),
            left_body=gcfg.get("left_body", "finger_left"),
            right_body=gcfg.get("right_body", "finger_right"),
        )

        # 任务（可选）
        self.task = task
        if self.task is not None:
            self.task.attach(self)

        # 位姿提供者：默认从 config.pose.source 读取，也可外部注入（测试/视觉算法）
        if pose_provider is not None:
            self.pose_provider = pose_provider
        else:
            src = pose_source or self.cfg.get("pose", {}).get("source", "gt")
            self.pose_provider = make_pose_provider(src, self)
        self.pose_provider.attach(self)

        # 初始零力矩
        self.n_substeps = self.cfg["sim"]["n_substeps"]
        self.home_qpos = np.array(self.ur5e_cfg["joints"]["home_qpos"], dtype=float)

        # 每步物理推进后、obs 构建前的回调（如抓取耦合更新枪位姿）
        self._post_step_hooks: list = []

    # ---------- actuator 收集 ----------
    def _collect_actuators(self, prefix: str) -> dict:
        """按 name 前缀筛选 actuator，返回 {关节名: actuator_id}。

        key 用 actuator 所驱动关节的名称（而非 actuator name 去前缀），
        这样 actuator name 如何命名都不影响查找。
        """
        out = {}
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name and name.startswith(prefix):
                jid = self.model.actuator_trnid[i, 0]
                jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                out[jname] = i
        return out

    # ---------- obs ----------
    def _build_obs(self) -> np.ndarray:
        qpos = self.data.qpos[self.arm_qposadr].copy()
        qvel = self.data.qvel[self.arm_dofadr].copy()
        # 末端位姿走 provider（GT 或视觉算法），与 task/IK 数据源统一
        ee_pos, ee_mat = self.pose_provider.get_ee_pose()
        gripper_open = np.array([self.gripper.get_opening()])
        obs = np.concatenate([qpos, qvel, ee_pos, ee_mat.flatten(), gripper_open])
        return obs.astype(np.float32)

    @property
    def obs_dim(self) -> int:
        return 6 + 6 + 3 + 9 + 1  # =25

    @property
    def action_dim(self) -> int:
        return self.arm_dof + 1  # 6 arm + 1 gripper

    # ---------- 生命周期 ----------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        # 设 home 位姿
        self.data.qpos[self.arm_qposadr] = self.home_qpos
        # 夹爪初始张开：qpos_l=range_l[0]=-0.025, qpos_r=range_r[1]=0.025（两指外扩）
        self.data.qpos[self.gripper.qposadr_l] = self.gripper.range_l[0]
        self.data.qpos[self.gripper.qposadr_r] = self.gripper.range_r[1]
        mujoco.mj_forward(self.model, self.data)
        # reset 后刷新 provider 缓存（视觉算法可在此跑首帧检测）
        self.pose_provider.update()
        if self.task is not None:
            self.task.reset(seed=seed)
        return self._build_obs()

    def step(self, action: np.ndarray) -> StepResult:
        action = np.asarray(action, dtype=float).reshape(-1)
        arm_action = action[:self.arm_dof]
        grip_ratio = float(action[self.arm_dof]) if action.size > self.arm_dof else 1.0

        self._apply_action(arm_action, grip_ratio)

        for _ in range(self.n_substeps):
            # 重力补偿前馈：将 qfrc_bias 注入 qfrc_applied，抵消重力造成的稳态误差
            if self.gravity_comp:
                mujoco.mj_forward(self.model, self.data)
                self.data.qfrc_applied[self.arm_dofadr] = self.data.qfrc_bias[self.arm_dofadr]
            else:
                self.data.qfrc_applied[self.arm_dofadr] = 0.0
            mujoco.mj_step(self.model, self.data)

        # 物理推进后刷新位姿提供者缓存（视觉算法在此跑检测，GT 无操作）
        self.pose_provider.update()

        # 执行每步后回调（抓取耦合在此更新被夹物体位姿并 mj_forward）
        for fn in self._post_step_hooks:
            fn()

        obs = self._build_obs()
        reward, done, info = 0.0, False, {}
        if self.task is not None:
            reward, done, info = self.task.step(obs)
        return StepResult(obs, reward, done, info)

    def _apply_action(self, arm_action: np.ndarray, grip_ratio: float):
        # 先把所有 ctrl 置零
        self.data.ctrl[:] = 0.0

        for name, jid in zip(self.arm_joint_names, self.arm_jids):
            self.data.ctrl[self.act_act_ids[name]] = arm_action[self.arm_joint_names.index(name)]

        # 夹爪用 position 控制；rl_grasp_joint 由 equality 镜像跟随，无需单独 actuator
        t_l, t_r = self.gripper.set_target(grip_ratio)
        if self.gripper.left_joint in self.act_act_ids:
            self.data.ctrl[self.act_act_ids[self.gripper.left_joint]] = t_l

    # ---------- 辅助 ----------
    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (pos[3], rot_mat[3,3])，来源取决于 pose_provider。"""
        return self.pose_provider.get_ee_pose()

    def body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """返回指定 body (pos[3], rot_mat[3,3])，来源取决于 pose_provider。"""
        return self.pose_provider.get_body_pose(name)

    def site_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """返回 site 的世界位姿 (pos[3], rot_mat[3,3])。直接读 MuJoCo 派生量。"""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError(f"site '{name}' not found")
        pos = self.data.site_xpos[sid].copy()
        mat = self.data.site_xmat[sid].reshape(3, 3).copy()
        return pos, mat

    def _subtree_geoms(self, bid: int) -> set:
        """收集 body 及其全部后代 body 的 geom id 集合。

        body_geomadr/body_geomnum 只覆盖直属 geom；枪头圆盘在子 body
        gun_pan、充电插座弹片环在子 body 内，接触/力查询必须用子树
        集合才能覆盖（否则静默漏检，导纳反馈失真）。
        """
        bodies = {bid}
        # 按 id 升序单轮扩张即可：MuJoCo 中 parent id 恒小于 child id，
        # 祖先必先于后代进入集合
        for k in range(self.model.nbody):
            if self.model.body_parentid[k] in bodies:
                bodies.add(k)
        geoms: set = set()
        for b in bodies:
            geoms.update(range(self.model.body_geomadr[b],
                               self.model.body_geomadr[b] + self.model.body_geomnum[b]))
        return geoms

    def body_collides_with(self, body_name: str, other_body_name: str) -> bool:
        """判断两 body（含各自子树）的 geom 是否当前存在接触。"""
        b1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        b2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, other_body_name)
        if b1 < 0 or b2 < 0:
            return False
        g1s = self._subtree_geoms(b1)
        g2s = self._subtree_geoms(b2)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if ((c.geom1 in g1s and c.geom2 in g2s) or
                    (c.geom2 in g1s and c.geom1 in g2s)):
                return True
        return False

    def geom_body_collides(self, geom_name: str, body_name: str) -> bool:
        """判断指定 geom 是否与某 body（含子树）的 geom 接触。"""
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

    def contact_force_between(self, body1_name: str, body2_name: str) -> np.ndarray:
        """计算 body1 受到的来自 body2 的接触力合力（世界系 [fx,fy,fz]）。

        双方均含子树 geom（枪的 gun_pan、插座的弹片子 body 都计入）。
        mj_contactForce 返回接触约束力（contact frame），contact.frame 的法向
        从 geom1 指向 geom2。R@f 是作用在 geom2 上的力（推开 geom2）。
        故需要根据 geom 归属决定符号，才能得到 body1 受力。
        """
        b1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
        b2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_name)
        if b1 < 0 or b2 < 0:
            return np.zeros(3)
        g1s = self._subtree_geoms(b1)
        g2s = self._subtree_geoms(b2)
        f_total = np.zeros(3)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            f = np.zeros(6)
            R = c.frame.reshape(3, 3)
            if c.geom1 in g1s and c.geom2 in g2s:
                # 法向 body1→body2，R@f 是 body2 受力；body1 受力 = -(R@f)
                mujoco.mj_contactForce(self.model, self.data, i, f)
                f_total -= R @ f[:3]
            elif c.geom2 in g1s and c.geom1 in g2s:
                # 法向 body2→body1，R@f 是 body1 受力
                mujoco.mj_contactForce(self.model, self.data, i, f)
                f_total += R @ f[:3]
        return f_total

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.data.qpos[self.arm_qposadr].copy(),
                self.data.qvel[self.arm_dofadr].copy())

    # ---------- 每步回调（抓取耦合等）----------
    def add_post_step_hook(self, fn):
        self._post_step_hooks.append(fn)

    def clear_post_step_hooks(self):
        self._post_step_hooks.clear()

    # ---------- viewer ----------
    def launch_viewer(self):
        """阻塞式启动交互 viewer（调试用）。"""
        import mujoco.viewer
        mujoco.viewer.launch(self.model, self.data)

    def launch_passive_viewer(self):
        """返回 passive viewer handle，可在循环中同步。"""
        import mujoco.viewer
        return mujoco.viewer.launch_passive(self.model, self.data)

    def render_markers(self, viewer):
        """在 viewer 中绘制 task.markers（红色目标点等），需在 viewer.sync() 前调用。

        利用 viewer.user_scn 添加自定义几何体，不修改模型、不影响物理。
        无 task 或无 markers 时清空 user_scn。
        """
        scn = viewer.user_scn
        scn.ngeom = 0
        markers = self.task.markers if self.task is not None else []
        for mk in markers:
            i = scn.ngeom
            if i >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(
                scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[mk.get("size", 0.02), 0, 0],
                pos=np.asarray(mk["pos"], float),
                mat=np.eye(3).flatten(),
                rgba=np.asarray(mk.get("rgba", [1, 0, 0, 1]), float),
            )
            scn.ngeom += 1
