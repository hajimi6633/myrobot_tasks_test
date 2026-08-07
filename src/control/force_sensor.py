"""六维力传感器：读取 MuJoCo force/torque sensor 数据。

MuJoCo 的 force/torque sensor 读取 site 所在 body subtree 的合力/合力矩
（含重力、接触力、约束力）。对于 free-joint body，在静平衡时合力为 0
（重力与约束/接触力相互抵消）。当有额外接触（如插枪碰壁）时，传感器
读数反映该额外接触力。因此直接返回原始读数（世界系），不做重力补偿。
"""
from __future__ import annotations
import numpy as np
import mujoco


class ForceTorqueSensor:
    """封装一个 site 上的 force/torque sensor，返回世界系六维力。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, site_name: str):
        self.model = model
        self.data = data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"site '{site_name}' not found")
        self.force_adr, self.torque_adr = self._find_sensor_adr(site_name)
        self.body_id = model.site_bodyid[self.site_id]
        self.subtree_mass = float(model.body_subtreemass[self.body_id])

    def _find_sensor_adr(self, site_name: str):
        """按 sensor 关联的 site 名查找 force/torque 的 sensordata 地址。"""
        force_adr = torque_adr = None
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if not name:
                continue
            stype = self.model.sensor_type[i]
            sid = self.model.sensor_objid[i]
            if sid != self.site_id:
                continue
            if stype == mujoco.mjtSensor.mjSENS_FORCE:
                force_adr = int(self.model.sensor_adr[i])
            elif stype == mujoco.mjtSensor.mjSENS_TORQUE:
                torque_adr = int(self.model.sensor_adr[i])
        return force_adr, torque_adr

    def read(self) -> np.ndarray:
        """返回世界系六维力 [fx,fy,fz,tx,ty,tz]（原始读数，不做重力补偿）。

        对于 free-joint body，合力 = 重力 + 接触力 + 约束力。
        静平衡时合力 ≈ 0；有额外接触时合力反映接触力。
        """
        if self.force_adr is None or self.torque_adr is None:
            return np.zeros(6)
        f_local = self.data.sensordata[self.force_adr:self.force_adr + 3].copy()
        t_local = self.data.sensordata[self.torque_adr:self.torque_adr + 3].copy()
        R = self.data.site_xmat[self.site_id].reshape(3, 3)
        f_world = R @ f_local
        t_world = R @ t_local
        return np.concatenate([f_world, t_world])

    def force(self) -> np.ndarray:
        """世界系合力 [fx,fy,fz]。"""
        return self.read()[:3]

    def force_magnitude(self) -> float:
        """合力大小 (N)。"""
        return float(np.linalg.norm(self.force()))

    def is_available(self) -> bool:
        return self.force_adr is not None and self.torque_adr is not None
