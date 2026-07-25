"""启动交互 viewer 检查模型。用法: python -m scripts.visualize"""
from __future__ import annotations
from src.env.arm_env import ArmEnv


def main():
    env = ArmEnv(action_mode="position")
    env.reset()
    print("Viewer 启动中，关闭窗口退出...")
    env.launch_viewer()


if __name__ == "__main__":
    main()
