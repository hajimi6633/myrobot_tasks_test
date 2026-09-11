"""rclike 基础设施：仿真时钟、节点日志、参数管理。"""
from __future__ import annotations


class SimClock:
    """仿真时钟：由 Executor 在每个控制拍后推进，全局唯一时间源。

    旁路线程（渲染/视觉/安全）只读 now()，用于打时间戳和新鲜度判断。
    仿真时间（而非墙钟）是全链路时间戳的基准，保证可复现。
    """

    def __init__(self):
        self._t = 0.0

    def advance(self, dt: float) -> None:
        self._t += dt

    @property
    def now(self) -> float:
        return self._t


class Logger:
    """节点级日志：[等级][节点名] 前缀。替代脚本里散落的 print 诊断。"""

    LEVELS = ("debug", "info", "warn", "error")

    def __init__(self, node_name: str, level: str = "info"):
        self.name = node_name
        self.level = self.LEVELS.index(level)

    def _log(self, lv: int, msg: str):
        if lv >= self.level:
            print(f"[{self.LEVELS[lv][0].upper()}][{self.name}] {msg}")

    def debug(self, msg: str): self._log(0, msg)
    def info(self, msg: str): self._log(1, msg)
    def warn(self, msg: str): self._log(2, msg)
    def error(self, msg: str): self._log(3, msg)


class ParameterStore:
    """参数声明 + 默认值 + 覆盖。

    替代散落常量：如 charging_phases 里的 F_BLOCK / INSERT_STEP / ALIGN_TOL，
    全部改为节点内 declare_parameter，launch 时用 YAML 覆盖。
    """

    def __init__(self):
        self._params: dict[str, tuple] = {}   # name -> (value, desc)

    def declare(self, name: str, default, desc: str = ""):
        if name in self._params:
            raise KeyError(f"参数重复声明: {name}")
        self._params[name] = (default, desc)
        return default

    def get(self, name):
        if name not in self._params:
            raise KeyError(f"未声明参数: {name}")
        return self._params[name][0]

    def override(self, name: str, value):
        """launch 阶段用 YAML 值覆盖默认值（仅允许覆盖已声明参数，防拼写错误）。"""
        if name not in self._params:
            raise KeyError(f"覆盖了未声明的参数: {name}")
        self._params[name] = (value, self._params[name][1])

    def to_dict(self) -> dict:
        return {k: v[0] for k, v in self._params.items()}
