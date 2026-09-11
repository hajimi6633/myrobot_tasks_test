"""运行入口：组装 charging 栈并运行。

用法:
  python -m scripts.run_stack --ticks 200            # 裸跑（无视觉）
  python -m scripts.run_stack --ticks 2000 --vision  # 启用视觉 worker
"""
from __future__ import annotations
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="models/scene_table.xml")
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--vision", action="store_true",
                    help="启用视觉节点（需先实现 VisionNode._detect）")
    args = ap.parse_args()

    from rcs.launch import build_charging_stack
    ex, h = build_charging_stack(args.scene, vision=args.vision)
    h["task"].send_goal()
    ex.spin(n_ticks=args.ticks)


if __name__ == "__main__":
    main()
