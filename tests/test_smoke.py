"""基础冒烟测试：模型加载、env step、IK 收敛、位姿源切换。"""
import numpy as np
import mujoco

from src.env.arm_env import ArmEnv
from src.control.ik_solver import IKSolver
from src.env.tasks.reach import ReachTask
from src.perception.pose_provider import (
    GroundTruthPoseProvider, VisionPoseProvider, make_pose_provider,
)


def test_model_loads():
    # position 模式：6 个臂 position + 1 个夹爪 position 执行器
    env = ArmEnv(action_mode="position")
    assert env.model.nq >= 8  # 6 arm + 2 finger + freejoint
    assert env.model.nu == 7, f"position 模式 nu 应为 7，实际 {env.model.nu}"
    # actuator 收集：只有 act_ 前缀
    assert len(env.act_act_ids) == 7
    assert "ll_grasp_joint" in env.act_act_ids
    print("test_model_loads OK")


def test_reset_step():
    env = ArmEnv(action_mode="position")
    obs = env.reset(seed=42)
    assert obs.shape == (env.obs_dim,)
    action = np.concatenate([env.home_qpos, [1.0]])
    res = env.step(action)
    assert res.obs.shape == obs.shape
    print("test_reset_step OK")


def test_ik_position_mode():
    env = ArmEnv(action_mode="position")
    env.reset(seed=0)
    ik = IKSolver(env, max_iter=50, tol=1e-3)
    target = np.array([0.2, 0.4, 0.6])
    q = ik.solve(target)
    # IK 不污染仿真状态：用返回的 q 重新设置并 forward 验证
    env.data.qpos[env.arm_qposadr] = q
    mujoco.mj_forward(env.model, env.data)
    ee = env.data.site_xpos[env.ee_site_id]
    err = np.linalg.norm(ee - target)
    assert err < 0.05, f"IK 误差过大: {err}"
    print(f"test_ik_position_mode OK, err={err:.4f}")


def test_task_reach():
    env = ArmEnv(action_mode="position", task=ReachTask())
    env.reset(seed=1)
    # 末端朝目标方向给动作
    q_des = env.home_qpos + 0.1
    res = env.step(np.concatenate([q_des, [1.0]]))
    assert "dist" in res.info
    print(f"test_task_reach OK, dist={res.info['dist']:.3f}")


def test_pose_source_gt():
    """GT 模式：ee_pose 应等于 MuJoCo site 真值。"""
    env = ArmEnv(action_mode="position", pose_source="gt")
    env.reset(seed=0)
    pos, rot = env.ee_pose()
    gt_pos = env.data.site_xpos[env.ee_site_id]
    assert np.allclose(pos, gt_pos), "GT ee_pose 与真值不一致"
    # body_pose 也应等于真值
    opos, _ = env.body_pose("object")
    oid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "object")
    assert np.allclose(opos, env.data.xpos[oid])
    print("test_pose_source_gt OK")


def test_pose_source_vision_fallback():
    """Vision 模式未实现检测时，应回退到 GT，输出与 GT 一致。"""
    env = ArmEnv(action_mode="position", pose_source="vision")
    assert isinstance(env.pose_provider, VisionPoseProvider)
    env.reset(seed=0)
    pos_v, _ = env.ee_pose()
    gt_pos = env.data.site_xpos[env.ee_site_id]
    assert np.allclose(pos_v, gt_pos), "Vision fallback 输出与 GT 不一致"
    # step 后仍应一致（fallback 链路工作）
    env.step(np.concatenate([env.home_qpos, [1.0]]))
    pos_v2, _ = env.ee_pose()
    assert np.allclose(pos_v2, env.data.site_xpos[env.ee_site_id])
    print("test_pose_source_vision_fallback OK")


def test_pose_provider_injection():
    """外部注入自定义 provider：注入 GT 后 obs 中 ee 应来自 provider。"""

    class NoisyProvider(GroundTruthPoseProvider):
        """演示用：末端位置加固定偏移，验证 obs 确实走 provider。"""
        OFFSET = np.array([0.1, 0.2, 0.3])

        def get_ee_pose(self):
            pos, rot = super().get_ee_pose()
            return pos + self.OFFSET, rot

    env = ArmEnv(action_mode="position",
                 pose_provider=NoisyProvider())
    env.reset(seed=0)
    pos, _ = env.ee_pose()
    gt_pos = env.data.site_xpos[env.ee_site_id]
    assert np.allclose(pos, gt_pos + NoisyProvider.OFFSET), "自定义 provider 未生效"
    # obs 中的 ee 段也应带偏移
    obs = env._build_obs()
    obs_ee = obs[12:15]
    assert np.allclose(obs_ee, gt_pos + NoisyProvider.OFFSET)
    print("test_pose_provider_injection OK")


if __name__ == "__main__":
    test_model_loads()
    test_reset_step()
    test_ik_position_mode()
    test_task_reach()
    test_pose_source_gt()
    test_pose_source_vision_fallback()
    test_pose_provider_injection()
    print("\nALL TESTS PASSED")
