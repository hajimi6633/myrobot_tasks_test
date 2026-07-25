"""配置加载工具：相对项目根定位 config 与 models 路径。"""
from __future__ import annotations
import os
from pathlib import Path
import yaml

# 项目根：src/config.py 的上两级
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(rel: str) -> str:
    """把相对项目根的路径转成绝对路径；已是绝对路径（含 Windows 盘符）则原样返回。"""
    if os.path.isabs(rel) or (len(rel) >= 2 and rel[1] == ':'):
        return rel
    return str(PROJECT_ROOT / rel)


def load_yaml(rel: str) -> dict:
    with open(project_path(rel), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
