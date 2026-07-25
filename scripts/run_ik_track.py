"""轨迹跟踪测试：验证 IK 算法正确性。

机械臂初始姿态为 home_qpos [-1.5708, 0, 1.5708, 0, 1.5708, 0.0]，
通过 IK 让末端沿指定形状轨迹运动，输出末端实际跟踪轨迹，
对比期望轨迹，判定 IK 是否正确。

支持轨迹（--traj）:
  line     沿 Y 轴直线平移（默认）
  circle   XY 平面圆，圆心在 p0 正前方 scale 处，半径 scale
  square   正方形（边长 2*scale），起点 p0，依次 +Y → +X → -Y → -X
  star     5 角星（外径 scale，内径 scale*0.382），起点为顶部外顶点

用法:
  python -m scripts.run_ik_track                       # line, 无 viewer
  python -m scripts.run_ik_track --traj circle          # 圆形
  python -m scripts.run_ik_track --traj star --viewer   # 星形 + viewer
"""
from __future__ import annotations
import argparse
import numpy as np
import mujoco
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from src.env.arm_env import ArmEnv
from src.control.ik_solver import IKSolver


# ---------- 期望轨迹生成器 ----------
def _min_jerk_s(N: int) -> np.ndarray:
    """归一化 min-jerk 进度 s(t)∈[0,1]，起止速度/加速度为 0。"""
    t = np.linspace(0, 1, N)
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def gen_line(p0: np.ndarray, scale: float, N: int) -> np.ndarray:
    """沿 Y 轴直线平移 scale，min-jerk 位置插值。"""
    s = _min_jerk_s(N)
    end = p0.copy(); end[1] += scale
    return p0[None, :] + (end - p0)[None, :] * s[:, None]


def gen_circle(p0: np.ndarray, scale: float, N: int) -> np.ndarray:
    """XY 平面圆：圆心在 p0 正前方 scale 处，半径 scale，起点 p0。"""
    s = _min_jerk_s(N)
    theta = -np.pi / 2 + 2 * np.pi * s          # s=0 时在圆底（=p0）
    center = p0 + np.array([0, scale, 0])
    pos = np.zeros((N, 3))
    pos[:, 0] = center[0] + scale * np.cos(theta)
    pos[:, 1] = center[1] + scale * np.sin(theta)
    pos[:, 2] = p0[2]
    return pos


def _interp_polyline(verts: np.ndarray, N: int) -> np.ndarray:
    """沿闭合多边形顶点 verts (M,3)（首尾相同）线性插值，min-jerk 时间。"""
    s = _min_jerk_s(N)
    n_edges = len(verts) - 1
    seg = np.clip((s * n_edges).astype(int), 0, n_edges - 1)
    frac = s * n_edges - seg
    return verts[seg] + (verts[seg + 1] - verts[seg]) * frac[:, None]


def gen_square(p0: np.ndarray, scale: float, N: int) -> np.ndarray:
    """正方形（边长 2*scale），起点 p0，依次 +Y → +X → -Y → -X 回到 p0。"""
    L = 2 * scale
    verts = np.array([
        p0,
        p0 + [0, L, 0],
        p0 + [L, L, 0],
        p0 + [L, 0, 0],
        p0,
    ], dtype=float)
    return _interp_polyline(verts, N)


def gen_star(p0: np.ndarray, scale: float, N: int) -> np.ndarray:
    """5 角星（外径 scale，内径 scale*0.382），起点为顶部外顶点，顺时针绕一周。"""
    R, r = scale, scale * 0.382
    n_edges = 10  # 5 外 + 5 内 交替，共 10 条边
    angles = np.pi / 2 - np.arange(n_edges) * (2 * np.pi / n_edges)  # 顺时针
    radii = np.array([R if i % 2 == 0 else r for i in range(n_edges)])
    verts = np.zeros((n_edges + 1, 3))
    verts[:n_edges, 0] = radii * np.cos(angles)
    verts[:n_edges, 1] = radii * np.sin(angles)
    verts[n_edges] = verts[0]                   # 闭合
    verts += p0
    return _interp_polyline(verts, N)


TRAJ_FACTORIES = {
    "line": gen_line,
    "circle": gen_circle,
    "square": gen_square,
    "star": gen_star,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--T", type=float, default=4.0, help="轨迹总时长 (s)")
    ap.add_argument("--dt", type=float, default=0.02, help="控制周期 (s)")
    ap.add_argument("--traj", choices=list(TRAJ_FACTORIES), default="line",
                    help="轨迹形状: line/circle/square/star (默认 line)")
    ap.add_argument("--scale", type=float, default=0.20,
                    help="轨迹尺度 (m)：line=平移量, circle=半径, square=半边长, star=外径")
    ap.add_argument("--no-gravity-comp", action="store_true",
                    help="关闭 qfrc_bias 重力补偿前馈（默认开启）")
    ap.add_argument("--save", type=str, default=None,
                    help="将对比图保存到指定路径 (PNG)；不传则不保存")
    ap.add_argument("--show", action="store_true",
                    help="交互式显示图片（需要图形环境）")
    args = ap.parse_args()

    env = ArmEnv(action_mode="position", gravity_comp=not args.no_gravity_comp)
    ik = IKSolver(env, max_iter=200, tol=1e-5)
    obs = env.reset(seed=0)

    print(f"重力补偿前馈 (qfrc_bias): {'开启' if env.gravity_comp else '关闭'}")
    print(f"轨迹形状 / 尺度: {args.traj} / {args.scale*100:.1f} cm")

    # 确认初始姿态
    q0 = env.data.qpos[env.arm_qposadr].copy()
    print(f"初始关节角: {q0}")
    mujoco.mj_forward(env.model, env.data)
    p0 = env.data.site_xpos[env.ee_site_id].copy()
    print(f"初始末端位置: {p0}\n")

    N = int(args.T / args.dt)
    # 期望末端轨迹 (N, 3)
    desired_traj = TRAJ_FACTORIES[args.traj](p0, args.scale, N)
    target_end = desired_traj[-1]

    viewer = env.launch_passive_viewer() if args.viewer else None

    # 记录实际末端位置、关节角、IK 误差
    actual_traj = np.zeros((N, 3))
    ik_errs = np.zeros(N)
    joint_traj = np.zeros((N, 6))

    q_cur = q0.copy()
    for i in range(N):
        # IK 求解当前期望末端位置对应的关节角
        target_pos = desired_traj[i]
        q_des = ik.solve(target_pos, q_init=q_cur)
        ik_err = np.linalg.norm(
            _ee_at(env, q_des) - target_pos)
        ik_errs[i] = ik_err
        q_cur = q_des.copy()

        # 执行（position 模式直接写 ctrl）
        action = np.concatenate([q_des, [1.0]])  # 夹爪张开
        env.step(action)
        mujoco.mj_forward(env.model, env.data)

        actual_traj[i] = env.data.site_xpos[env.ee_site_id].copy()
        joint_traj[i] = env.data.qpos[env.arm_qposadr].copy()

        if viewer is not None:
            viewer.sync()

    # ---- 结果分析 ----
    tracking_err = np.linalg.norm(actual_traj - desired_traj, axis=1)
    print("=" * 60)
    print("轨迹跟踪结果")
    print("=" * 60)
    print(f"步数 N            : {N}")
    print(f"控制周期 dt       : {args.dt} s")
    print(f"轨迹总时长 T      : {args.T} s")
    print(f"轨迹尺度          : {args.scale*100:.1f} cm")
    print()
    print(f"IK 求解误差  max  : {ik_errs.max():.6f} m")
    print(f"IK 求解误差  mean : {ik_errs.mean():.6f} m")
    print()
    print(f"跟踪误差    max  : {tracking_err.max():.6f} m")
    print(f"跟踪误差    mean : {tracking_err.mean():.6f} m")
    print()
    # 直线度：仅对 line 轨迹有意义（X、Z 方向应保持恒定）
    if args.traj == "line":
        line_err_x = np.abs(actual_traj[:, 0] - p0[0])
        line_err_z = np.abs(actual_traj[:, 2] - p0[2])
        print(f"直线度偏差  X max: {line_err_x.max():.6f} m")
        print(f"直线度偏差  Z max: {line_err_z.max():.6f} m")
        print()
    print(f"末端起点   : {actual_traj[0]}")
    print(f"末端终点   : {actual_traj[-1]}")
    print(f"期望终点   : {target_end}")
    print(f"终点误差   : {np.linalg.norm(actual_traj[-1] - target_end):.6f} m")

    # 判定：IK 误差 + 跟踪误差；line 模式额外检查直线度
    print("\n" + "=" * 60)
    ok = ik_errs.max() < 1e-3 and tracking_err.max() < 5e-3
    if args.traj == "line":
        ok = ok and line_err_x.max() < 5e-3 and line_err_z.max() < 5e-3
    print(f"判定: {'✓ IK 正确，轨迹跟踪良好' if ok else '✗ IK 或跟踪存在问题'}")
    print("=" * 60)

    # 输出每 10% 进度的采样（末位置范数 + 跟踪误差）
    print("\n采样点（每 10% 进度）:")
    print(f"{'进度':6s} | {'期望位置':28s} | {'实际位置':28s} | {'IK 误差':9s} | {'跟踪误差':9s}")
    for pct in range(0, 101, 10):
        idx = min(int(pct / 100 * (N - 1)), N - 1)
        dp = desired_traj[idx]
        ap = actual_traj[idx]
        print(f"{pct:5d}% | [{dp[0]:7.4f} {dp[1]:7.4f} {dp[2]:7.4f}] | "
              f"[{ap[0]:7.4f} {ap[1]:7.4f} {ap[2]:7.4f}] | "
              f"{ik_errs[idx]:9.6f} | {tracking_err[idx]:9.6f}")

    # ---- 绘图 ----
    _plot_results(desired_traj, actual_traj, tracking_err, ik_errs,
                  args.traj, args.scale, args.T, args.dt,
                  save_path=args.save, show=args.show)


def _plot_results(desired, actual, tracking_err, ik_errs,
                  traj_name: str, scale: float, T: float, dt: float,
                  save_path: str | None = None, show: bool = False):
    """绘制 3 子图：XY 俯视轨迹对比、3D 轨迹对比、跟踪误差曲线。"""
    t = np.arange(len(tracking_err)) * dt  # 时间轴 (s)
    fig = plt.figure(figsize=(15, 5))
    suptitle = (f"轨迹跟踪对比  [{traj_name}, scale={scale*100:.1f}cm, "
                f"T={T}s, dt={dt}s]")
    fig.suptitle(suptitle, fontsize=13)

    # ---- 子图1：XY 俯视（所有轨迹均在 XY 平面）----
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(desired[:, 0], desired[:, 1], 'b--', linewidth=1.5, label='期望')
    ax1.plot(actual[:, 0], actual[:, 1], 'r-', linewidth=1.2, label='实际')
    ax1.plot(desired[0, 0], desired[0, 1], 'go', markersize=7, label='起点')
    ax1.plot(desired[-1, 0], desired[-1, 1], 'k^', markersize=7, label='终点')
    ax1.set_aspect('equal')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('XY 俯视轨迹')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # ---- 子图2：3D 轨迹对比 ----
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.plot(desired[:, 0], desired[:, 1], desired[:, 2], 'b--',
             linewidth=1.5, label='期望')
    ax2.plot(actual[:, 0], actual[:, 1], actual[:, 2], 'r-',
             linewidth=1.2, label='实际')
    ax2.scatter(*desired[0], color='g', s=40, label='起点')
    ax2.scatter(*desired[-1], color='k', marker='^', s=40, label='终点')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('3D 轨迹')
    ax2.legend(fontsize=8)

    # ---- 子图3：跟踪误差 + IK 误差 vs 时间 ----
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(t, tracking_err * 1000, 'r-', linewidth=1.2, label='跟踪误差')
    ax3.plot(t, ik_errs * 1000, 'b-', linewidth=1.0, alpha=0.6, label='IK 误差')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('误差 (mm)')
    ax3.set_title('误差曲线')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"\n图片已保存: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def _ee_at(env, q):
    """在给定关节角下计算末端位置（不修改仿真状态）。"""
    backup = env.data.qpos.copy()
    env.data.qpos[env.arm_qposadr] = q
    mujoco.mj_forward(env.model, env.data)
    pos = env.data.site_xpos[env.ee_site_id].copy()
    env.data.qpos[:] = backup
    mujoco.mj_forward(env.model, env.data)
    return pos


if __name__ == "__main__":
    main()
