#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从100米风场计算60米风场

使用风廓线理论（幂律或对数律方法）进行高度订正
"""

import numpy as np
from typing import Tuple


class WindHeightConverter:
    """风场高度转换器 - 从100米计算60米风场"""

    # 风廓线参数
    WIND_PROFILE_PARAMS = {
        "power_law_exponent": 0.143,      # 幂律指数，适用于中性大气
        "reference_height": 100.0,        # 参考高度（米）
        "target_height": 60.0,            # 目标高度（米）
        "log_law_roughness": 0.03        # 对数律粗糙度（米），适用于平坦地形
    }

    @staticmethod
    def calculate_wind_speed_direction(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算风速风向

        Parameters
        ----------
        u : np.ndarray
            U分量 (m/s)
        v : np.ndarray
            V分量 (m/s)

        Returns
        -------
        tuple
            (风速, 风向)
        """
        # 计算风速
        wspd = np.sqrt(u ** 2 + v ** 2)

        # 计算风向 (气象标准：0°=北风，90°=东风，180°=南风，270°=西风)
        wdir = (np.degrees(np.arctan2(u, v)) + 180) % 360

        return wspd, wdir

    @staticmethod
    def wspd_wdir_to_uv(wspd: np.ndarray, wdir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将风速风向转换为U/V分量

        Parameters
        ----------
        wspd : np.ndarray
            风速 (m/s)
        wdir : np.ndarray
            风向 (度，0°=北风)

        Returns
        -------
        tuple
            (U分量, V分量)
        """
        # 转换为弧度
        wdir_rad = np.radians(wdir)

        # 计算U/V分量
        # 注意：气象风向定义中，U正方向为西风，V正方向为南风
        u = -wspd * np.sin(wdir_rad)
        v = -wspd * np.cos(wdir_rad)

        return u, v

    @staticmethod
    def calculate_60m_wind_from_100m(u_100m: np.ndarray, v_100m: np.ndarray,
                                     method: str = 'power_law') -> Tuple[np.ndarray, np.ndarray]:
        """
        从100米风场计算60米风场

        Parameters
        ----------
        u_100m : np.ndarray
            100米U分量 (m/s)
        v_100m : np.ndarray
            100米V分量 (m/s)
        method : str
            计算方法: 'power_law' (幂律) 或 'log_law' (对数律)

        Returns
        -------
        tuple
            (60米U分量, 60米V分量)
        """
        # 获取参数
        params = WindHeightConverter.WIND_PROFILE_PARAMS
        z_ref = params["reference_height"]
        z_target = params["target_height"]

        # 计算风速
        wspd_100m, wdir_100m = WindHeightConverter.calculate_wind_speed_direction(u_100m, v_100m)

        if method == 'power_law':
            # 幂律方法
            # 公式: V2 = V1 × (z2/z1)^α
            # 其中 α 为幂律指数（通常0.1-0.2，这里取0.143）
            alpha = params["power_law_exponent"]
            wspd_60m = wspd_100m * (z_target / z_ref) ** alpha

        elif method == 'log_law':
            # 对数律方法
            # 公式: V2 = V1 × ln(z2/z0) / ln(z1/z0)
            # 其中 z0 为地表粗糙度（米）
            z0 = params["log_law_roughness"]
            wspd_60m = wspd_100m * (np.log(z_target / z0) / np.log(z_ref / z0))
        else:
            raise ValueError(f"未知的计算方法: {method}")

        # 风向通常随高度变化不大，使用相同的风向
        wdir_60m = wdir_100m.copy()

        # 将60米风速风向转换回U/V分量
        u_60m, v_60m = WindHeightConverter.wspd_wdir_to_uv(wspd_60m, wdir_60m)

        return u_60m, v_60m


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    u_100m = np.array([[5.0, 8.0], [3.0, 6.0]])  # 100米U分量
    v_100m = np.array([[4.0, 6.0], [2.0, 5.0]])  # 100米V分量

    # 计算原始风速
    wspd_100m, wdir_100m = WindHeightConverter.calculate_wind_speed_direction(u_100m, v_100m)
    print("100米风场:")
    print(f"风速 (m/s):\n{wspd_100m}")
    print(f"风向 (度):\n{wdir_100m}")

    # 使用幂律方法计算60米风场
    u_60m, v_60m = WindHeightConverter.calculate_60m_wind_from_100m(
        u_100m, v_100m, method='power_law'
    )
    wspd_60m, wdir_60m = WindHeightConverter.calculate_wind_speed_direction(
        u_60m, v_60m
    )
    print("\n60米风场 (幂律方法):")
    print(f"风速 (m/s):\n{wspd_60m}")
    print(f"风向 (度):\n{wdir_60m}")
    print(f"风速比 (60m/100m):\n{wspd_60m / wspd_100m}")
