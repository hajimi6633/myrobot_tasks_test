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

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.data.qpos[self.arm_qposadr].copy(),
                self.data.qvel[self.arm_dofadr].copy())

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
