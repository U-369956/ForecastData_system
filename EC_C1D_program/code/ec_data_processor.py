#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ECMWF多要素数据处理工具 - 从NetCDF文件开始处理不包含CDO转换步骤，专注于数据处理

批量处理 nc_output 目录下的所有NC文件
                                                                                                                                                  python3 /home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/ec_data_processor.py \
                                                                                                           --input-dir /home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/nc_output \
                                                                                                              --output-dir /home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/1KM1H \
                                                                                                                 --verbose
                                                                                                                                                                                           其他常用选项：
                                                                                                                                                                          # 只处理某个要素
                                                                                                                                                                         python3 /home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/ec_data_processor.py \
                                                                                                           --input-dir /home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/nc_output \      --element TEM \
                                                                                                                                                                          --output-dir /home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/1KM1H
                                                                                                               # 单个文件处理
                                                                                                               python3 /home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/ec_data_processor.py \
                                                                                                                   --element TEM \
                                                                                                                   --nc-file /home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/nc_output/ECMFC1D_TEM_1_2026022612_GLB_1.nc \
                                                                                                                   --base-time 2026022612
                                                                                                               # 不跳过已存在文件（强制重新处理）
                                                                                                               python3 ... --no-skip-existing
"""
import os
import sys
import logging
import argparse
import numpy as np
from netCDF4 import Dataset, date2num
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator, interp1d
import warnings
import time
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
import math
import pickle
import shutil


# ================= 配置类 =================
class Config:
    """集中管理所有配置参数"""

    # 区域范围
    REGION = {
        "lon_w": 96.0,
        "lon_e": 127.0,
        "lat_s": 34.0,
        "lat_n": 47.0
    }

    # 输出分辨率
    RESOLUTION = 0.01

    # 时区偏移（北京时）
    TIMEZONE_SHIFT = timedelta(hours=8)

    # 默认输出文件名格式
    OUTPUT_FILENAME_FORMAT = "{element}_0p01_1h_BJT_{time_str}.nc"

    # 要素配置
    ELEMENTS = {
        # 温度
        "TEM": {
            "description": "2m temperature",
            "grib_codes": {"value": "2t"},
            "requires_uv": False,
            "output_vars": ["temp"],
            "units": {"temp": "°C"},  # 修改为摄氏度
            "conversion": "K_to_C"  # 添加转换标志
        },
        # 10米风场
        "WIND_10M": {
            "description": "10m wind",
            "grib_codes": {"u": "10u", "v": "10v"},
            "requires_uv": True,
            "output_vars": ["WindSpeed10m", "WindDir10m"],
            "units": {"WindSpeed10m": "m s-1", "WindDir10m": "degree"},
            "height": 10
        },
        # 100米风场
        "WIND_100M": {
            "description": "100m wind",
            "grib_codes": {"u": "100u", "v": "100v"},
            "requires_uv": True,
            "output_vars": ["WindSpeed100m", "WindDir100m"],
            "units": {"WindSpeed100m": "m s-1", "WindDir100m": "degree"},
            "height": 100
        },
        # 60米风场（从100米计算）
        "WIND_60M": {
            "description": "60m wind (calculated from 100m)",
            "grib_codes": {"u": "100U", "v": "100V"},
            "requires_uv": True,
            "requires_calc": True,
            "output_vars": ["WindSpeed60m", "WindDir60m"],
            "units": {"WindSpeed60m": "m s-1", "WindDir60m": "degree"},
            "height": 60
        },
        # 10米U分量（单独输出）
        "10U": {
            "description": "10m u-component of wind",
            "grib_codes": {"value": "10u"},
            "requires_uv": False,
            "output_vars": ["u10"],
            "units": {"u10": "m s-1"},
            "conversion": None
        },
        # 10米V分量（单独输出）
        "10V": {
            "description": "10m v-component of wind",
            "grib_codes": {"value": "10v"},
            "requires_uv": False,
            "output_vars": ["v10"],
            "units": {"v10": "m s-1"},
            "conversion": None
        },
        # 100米U分量（单独输出）
        "100U": {
            "description": "100m u-component of wind",
            "grib_codes": {"value": "100u"},
            "requires_uv": False,
            "output_vars": ["u100"],
            "units": {"u100": "m s-1"},
            "conversion": None
        },
        # 100米V分量（单独输出）
        "100V": {
            "description": "100m v-component of wind",
            "grib_codes": {"value": "100v"},
            "requires_uv": False,
            "output_vars": ["v100"],
            "units": {"v100": "m s-1"},
            "conversion": None
        },
        # 阵风
        "GUST": {
            "description": "10m wind gust",
            "grib_codes": {"value": "10fg3"},
            "requires_uv": False,
            "output_vars": ["gust"],
            "units": {"gust": "m s-1"},
            "conversion": None
        },
        # 能见度
        "VIS": {
            "description": "visibility",
            "grib_codes": {"value": "vis"},
            "requires_uv": False,
            "output_vars": ["vis"],
            "units": {"vis": "m"},
            "conversion": None
        },
        # 气压
        "PRS": {
            "description": "surface pressure",
            "grib_codes": {"value": "msl"},
            "requires_uv": False,
            "output_vars": ["prs"],
            "units": {"prs": "Pa"},
            "conversion": None
        },
        # 降水
        "PRE": {
            "description": "total precipitation",
            "grib_codes": {"value": "tp"},
            "requires_uv": False,
            "output_vars": ["pre"],
            "units": {"pre": "mm"},
            "conversion": None
        },
        # 露点温度
        "DPT": {
            "description": "2m dewpoint temperature",
            "grib_codes": {"value": "2d"},
            "requires_uv": False,
            "output_vars": ["dpt"],
            "units": {"dpt": "°C"},  # 修改为摄氏度
            "conversion": "K_to_C"  # 添加转换标志
        },
        # 相对湿度
        "RH": {
            "description": "2m relative humidity",
            "grib_codes": {"temp": "TEM", "prs": "PRS", "dpt": "DPT"},
            "requires_uv": False,
            "requires_calc": True,
            "output_vars": ["rh"],
            "units": {"rh": "%"}
        },
        # MN2T6（过去6小时最低2m气温）
        "MN2T6": {
            "description": "minimum 2m temperature in past 6 hours",
            "grib_codes": {"value": "mn2t6"},
            "requires_uv": False,
            "output_vars": ["mn2t6"],
            "units": {"mn2t6": "°C"},
            "conversion": "K_to_C",
            "past_hours": 6
        },
        # MX2T6（过去6小时最高2m气温）
        "MX2T6": {
            "description": "maximum 2m temperature in past 6 hours",
            "grib_codes": {"value": "mx2t6"},
            "requires_uv": False,
            "output_vars": ["mx2t6"],
            "units": {"mx2t6": "°C"},
            "conversion": "K_to_C",
            "past_hours": 6
        },
        # TCC（总云量）
        "TCC": {
            "description": "total cloud cover",
            "grib_codes": {"value": "tcc"},
            "requires_uv": False,
            "output_vars": ["tcc"],
            "units": {"tcc": "%"},
            "conversion": None
        }
    }

    @classmethod
    def get_target_grid(cls) -> Tuple[np.ndarray, np.ndarray]:
        """生成目标网格 - 纬度升序（从南到北）"""
        lon_out = np.arange(cls.REGION["lon_w"],
                            cls.REGION["lon_e"] + cls.RESOLUTION / 2,
                            cls.RESOLUTION)
        # 纬度升序（从南到北）
        lat_out = np.arange(cls.REGION["lat_s"],
                            cls.REGION["lat_n"] + cls.RESOLUTION / 2,
                            cls.RESOLUTION)
        return lat_out, lon_out

    @classmethod
    def utc_to_bjt_str(cls, utc_time: datetime) -> str:
        """UTC时间转北京时字符串"""
        bjt_time = utc_time + cls.TIMEZONE_SHIFT
        return bjt_time.strftime("%Y%m%d%H")

    @classmethod
    def parse_time_from_filename(cls, filename: str) -> Optional[datetime]:
        """从文件名解析时间（支持多种格式）"""
        basename = os.path.basename(filename)
        # 尝试解析格式: ECMFC1D_TEM_1_2026010100_GLB_1.grib1
        if "ECMFC1D_" in basename:
            # 查找下划线分割的部分中包含数字的部分
            parts = basename.split('_')
            for part in parts:
                if len(part) >= 10 and part.isdigit():
                    try:
                        return datetime.strptime(part, "%Y%m%d%H")
                    except ValueError:
                        continue
        # 尝试直接查找连续的10位数字
        import re
        match = re.search(r'(\d{10})', basename)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y%m%d%H")
            except ValueError:
                pass
        return None


# ================= 气象计算工具类 =================
class MeteorologicalCalculator:
    """气象要素计算工具类"""

    @staticmethod
    def kelvin_to_celsius(kelvin_data: np.ndarray) -> np.ndarray:
        """开尔文转摄氏度"""
        return kelvin_data - 273.15

    @staticmethod
    def calculate_relative_humidity_from_temperature_dewpoint(temp_c: np.ndarray,
                                                           dpt_c: np.ndarray,
                                                           prs_pa: np.ndarray = None) -> np.ndarray:
        """
        从温度和露点温度计算相对湿度（精确版）
        使用Magnus公式计算饱和水汽压

        Parameters
        ----------
        temp_c : np.ndarray
            温度数据（摄氏度）
        dpt_c : np.ndarray
            露点温度数据（摄氏度）
        prs_pa : np.ndarray, optional
            气压数据（帕斯卡），如果不提供则使用海平面气压101325 Pa

        Returns
        -------
        np.ndarray
            相对湿度数据（百分比）
        """
        # 使用Magnus公式计算饱和水汽压（hPa）
        def calculate_saturation_vapor_pressure(temp):
            # Magnus公式参数
            a = 6.112
            b = 17.67
            c = 243.5
            # 计算饱和水汽压（hPa）
            es = a * np.exp((b * temp) / (c + temp))
            return es

        # 计算温度对应的饱和水汽压
        es_t = calculate_saturation_vapor_pressure(temp_c)
        # 计算露点温度对应的饱和水汽压
        es_d = calculate_saturation_vapor_pressure(dpt_c)
        # 如果没有提供气压，使用海平面标准气压
        if prs_pa is None:
            prs_pa = np.full_like(temp_c, 101325.0)  # Pa
        # 将气压转换为hPa
        prs_hpa = prs_pa / 100.0
        # 计算实际水汽压（从露点温度计算）
        e = es_d
        # 计算相对湿度
        rh = (e / es_t) * 100.0
        # 限制在0-100%范围内
        rh = np.clip(rh, 0.0, 100.0)
        # 处理NaN值：如果温度或露点有NaN，相对湿度也是NaN
        rh = np.where(np.isnan(temp_c) | np.isnan(dpt_c), np.nan, rh)
        return rh

    @staticmethod
    def calculate_relative_humidity_from_temperature(temp_c: np.ndarray) -> np.ndarray:
        """
        直接从温度计算相对湿度（简化版，作为后备）
        注意：这个方法不精确，建议使用基于露点温度的方法
        """
        # 简化算法：基于温度估算相对湿度
        rh = 50.0 + 30.0 * np.sin(temp_c / 10.0)  # 模拟相对湿度
        # 限制在0-100%范围内
        rh = np.clip(rh, 0.0, 100.0)
        # 处理NaN值
        rh = np.where(np.isnan(temp_c), np.nan, rh)
        return rh

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
        # 风场垂直外推参数（从EC_100mUV.py）
        z_ref = 100.0  # 参考高度（米）
        z_target = 60.0  # 目标高度（米）
        power_law_exponent = 0.143  # 幂律指数，适用于中性大气
        log_law_roughness = 0.03  # 对数律粗糙度（米），适用于平坦地形

        # 计算风速
        wspd_100m, wdir_100m = MeteorologicalCalculator.calculate_wind_speed_direction(u_100m, v_100m)

        if method == 'power_law':
            # 幂律方法
            wspd_60m = wspd_100m * (z_target / z_ref) ** power_law_exponent
        elif method == 'log_law':
            # 对数律方法
            wspd_60m = wspd_100m * (np.log(z_target / log_law_roughness) / np.log(z_ref / log_law_roughness))
        else:
            raise ValueError(f"未知的计算方法: {method}")

        # 风向通常随高度变化不大，使用相同的风向
        wdir_60m = wdir_100m.copy()

        # 将60米风速风向转换回U/V分量
        u_60m, v_60m = MeteorologicalCalculator.wspd_wdir_to_uv(wspd_60m, wdir_60m)

        return u_60m, v_60m

    @staticmethod
    def wspd_wdir_to_uv(wspd: np.ndarray, wdir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将风速风向转换为U/V分量

        Parameters
        ----------
        wspd : np.ndarray
            风速 (m/s)
        wdir : np.ndarray
            风向（度，0°=北风）

        Returns
        -------
        tuple
            (U分量, V分量)
        """
        # 转换为弧度
        wdir_rad = np.radians(wdir)
        # 计算U/V分量
        u = -wspd * np.sin(wdir_rad)  # 注意：气象风向定义中，U正方向为西风
        v = -wspd * np.cos(wdir_rad)  # V正方向为南风
        return u, v


# ================= MICAPS4格式保存类 =================
class MICAPS4Writer:
    """MICAPS第4类数据格式写入器 - 二进制格式"""

    # 要素到MICAPS要素名的映射
    ELEMENT_MAP = {
        "WIND_10M": "10WIND",
        "WIND_100M": "100WIND",
        "WIND_60M": "60WIND",
        "10U": "10UWND",
        "10V": "10VWND",
        "100U": "100UWND",
        "100V": "100VWND",
        "GUST": "10GUST",
        "VIS": "10VIS",
        "TEM": "2MT",
        "PRS": "SFC_PR",
        "DPT": "2MDPT",
        "RH": "2MRH",
        "PRE": "PRE",
        "MN2T6": "2MMN6T",
        "MX2T6": "2MMX6T"
    }

    @staticmethod
    def create_micaps4_filename(base_time: datetime, forecast_hour: int,
                                element: str = "WIND", model_name: str = "ECMWF",
                                timezone_shift: timedelta = None) -> str:
        """
        创建MICAPS4格式文件名

        Parameters
        ----------
        base_time : datetime
            预报起始时间（UTC）
        forecast_hour : int
            预报时效（小时）
        element : str
            要素名称
        model_name : str
            模式名称
        timezone_shift : timedelta, optional
            时区偏移

        Returns
        -------
        str
            MICAPS4格式文件名
        """
        if timezone_shift is None:
            timezone_shift = Config.TIMEZONE_SHIFT

        # 转换为本地时间
        local_time = base_time + timezone_shift
        # 格式: YYMMDDHH.TTT（年只取后2位，不包含要素名）
        # 例如: 26010108.000 表示2026年01月01日08时，预报时效000小时
        year_short = local_time.strftime('%y')  # 2位年份
        date_time = local_time.strftime('%m%d%H')  # 月日时
        return f"{year_short}{date_time}.{forecast_hour:03d}"

    @staticmethod
    def write_micaps4_scalar_file(data: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                                  base_time: datetime, forecast_hour: int,
                                  output_path: str, element: str = "TEM",
                                  model_name: str = "ECMWF", level: float = 1000.0,
                                  description: str = "", timezone_shift: timedelta = None) -> bool:
        """
        写入MICAPS4标量格式文件（type=4）

        Parameters
        ----------
        data : np.ndarray
            标量数据
        lats : np.ndarray
            纬度数组
        lons : np.ndarray
            经度数组
        base_time : datetime
            预报起始时间（UTC）
        forecast_hour : int
            预报时效
        output_path : str
            输出文件路径
        element : str
            要素名称
        model_name : str
            模式名称
        level : float
            层次（百帕）
        description : str
            数据描述
        timezone_shift : timedelta
            时区偏移

        Returns
        -------
        bool
            是否成功
        """
        try:
            if timezone_shift is None:
                timezone_shift = Config.TIMEZONE_SHIFT

            # 转换为本地时间
            local_time = base_time + timezone_shift

            # 数据检查
            if len(data.shape) != 2:
                raise ValueError(f"数据必须是2D数组，形状: {data.shape}")
            if data.shape != (len(lats), len(lons)):
                raise ValueError(f"数据形状{data.shape}与网格形状({len(lats)}, {len(lons)})不匹配")

            n_lat, n_lon = len(lats), len(lons)

            # 确保纬度从南到北（升序），经度从西到东（升序）
            if lats[0] > lats[-1]:  # 降序
                lats = lats[::-1]
                data = data[::-1, :]
            if lons[0] > lons[-1]:  # 降序
                lons = lons[::-1]
                data = data[:, ::-1]

            # 计算网格参数
            start_lat = float(lats[0])
            end_lat = float(lats[-1])
            start_lon = float(lons[0])
            end_lon = float(lons[-1])

            if n_lat > 1:
                lat_res = float(abs(lats[1] - lats[0]))
            else:
                lat_res = 0.0

            if n_lon > 1:
                lon_res = float(abs(lons[1] - lons[0]))
            else:
                lon_res = 0.0

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 写入二进制文件
            with open(output_path, 'wb') as f:
                # ============= 1. discriminator =============
                f.write(b'mdfs')

                # ============= 2. type: 4为模式标量数据 =============
                data_type = 4  # 标量网格数据
                f.write(np.int16(data_type).tobytes())

                # ============= 3. modelName =============
                model_name_bytes = model_name.upper().encode('ascii', 'ignore')[:20]
                if len(model_name_bytes) < 20:
                    model_name_bytes += b'\x00' * (20 - len(model_name_bytes))
                f.write(model_name_bytes)

                # ============= 4. element =============
                element_std = MICAPS4Writer.ELEMENT_MAP.get(element, element)
                element_bytes = element_std.encode('ascii', 'ignore')[:50]
                if len(element_bytes) < 50:
                    element_bytes += b'\x00' * (50 - len(element_bytes))
                f.write(element_bytes)

                # ============= 5. description =============
                desc = description or f"{element_std} data"
                if len(desc) > 30:
                    desc = desc[:30]
                try:
                    desc_bytes = desc.encode('gbk', 'ignore')[:30]
                except (UnicodeEncodeError, AttributeError):
                    desc_bytes = desc.encode('ascii', 'ignore')[:30]
                if len(desc_bytes) < 30:
                    desc_bytes += b'\x00' * (30 - len(desc_bytes))
                f.write(desc_bytes)

                # ============= 6. level =============
                f.write(np.float32(level).tobytes())

                # ============= 7-10. 起报日期和时间 =============
                # 注意：这里写入的是4位年份
                f.write(np.int32(local_time.year).tobytes())
                f.write(np.int32(local_time.month).tobytes())
                f.write(np.int32(local_time.day).tobytes())
                f.write(np.int32(local_time.hour).tobytes())

                # ============= 11. timezone =============
                timezone = 8  # 北京时区
                f.write(np.int32(timezone).tobytes())

                # ============= 12. period =============
                f.write(np.int32(forecast_hour).tobytes())

                # ============= 13-15. 经度范围信息 =============
                f.write(np.float32(start_lon).tobytes())
                f.write(np.float32(end_lon).tobytes())
                f.write(np.float32(lon_res).tobytes())

                # ============= 16. longitudeGridNumber =============
                f.write(np.int32(n_lon).tobytes())

                # ============= 17-19. 纬度范围信息 =============
                f.write(np.float32(start_lat).tobytes())
                f.write(np.float32(end_lat).tobytes())
                f.write(np.float32(lat_res).tobytes())

                # ============= 20. latitudeGridNumber =============
                f.write(np.int32(n_lat).tobytes())

                # ============= 21-23. 等值线相关信息 =============
                f.write(np.float32(0.0).tobytes())  # isolineStartValue
                f.write(np.float32(0.0).tobytes())  # isolineEndValue
                f.write(np.float32(0.0).tobytes())  # isolineSpace

                # ============= 24. Extent =============
                f.write(b'\x00' * 100)

                # ============= 数据区 =============
                # 处理NaN值
                data_clean = data.copy()
                nan_mask = np.isnan(data_clean)
                if np.any(nan_mask):
                    data_clean[nan_mask] = 9999.0

                # 展平数据（按MICAPS4要求的顺序：先纬向后经向）
                data_flat = data_clean.ravel(order='C').astype(np.float32)
                f.write(data_flat.tobytes())

            return True

        except Exception as e:
            return False

    @staticmethod
    def write_micaps4_vector_file(wspd: np.ndarray, wdir: np.ndarray,
                                  lats: np.ndarray, lons: np.ndarray,
                                  base_time: datetime, forecast_hour: int,
                                  output_path: str, model_name: str = "ECMWF",
                                  level: float = 1000.0,
                                  description: str = "10m wind speed and direction",
                                  timezone_shift: timedelta = None) -> bool:
        """
        写入MICAPS4矢量格式文件（type=11）- 风速和风向合并为一个矢量文件
        """
        try:
            if timezone_shift is None:
                timezone_shift = Config.TIMEZONE_SHIFT

            # 转换为本地时间
            local_time = base_time + timezone_shift

            # 数据检查
            if len(wspd.shape) != 2 or len(wdir.shape) != 2:
                raise ValueError(f"数据必须是2D数组，风速形状: {wspd.shape}, 风向形状: {wdir.shape}")
            if wspd.shape != wdir.shape:
                raise ValueError(f"风速和风向形状不匹配: {wspd.shape} != {wdir.shape}")
            if wspd.shape != (len(lats), len(lons)):
                raise ValueError(f"数据形状{wspd.shape}与网格形状({len(lats)}, {len(lons)})不匹配")

            n_lat, n_lon = len(lats), len(lons)

            # 确保纬度从南到北（升序），经度从西到东（升序）
            if lats[0] > lats[-1]:  # 降序
                lats = lats[::-1]
                wspd = wspd[::-1, :]
                wdir = wdir[::-1, :]
            if lons[0] > lons[-1]:  # 降序
                lons = lons[::-1]
                wspd = wspd[:, ::-1]
                wdir = wdir[:, ::-1]

            # 计算网格参数
            start_lat = float(lats[0])
            end_lat = float(lats[-1])
            start_lon = float(lons[0])
            end_lon = float(lons[-1])

            if n_lat > 1:
                lat_res = float(abs(lats[1] - lats[0]))
            else:
                lat_res = 0.0

            if n_lon > 1:
                lon_res = float(abs(lons[1] - lons[0]))
            else:
                lon_res = 0.0

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 写入二进制文件
            with open(output_path, 'wb') as f:
                # ============= 1. discriminator =============
                f.write(b'mdfs')

                # ============= 2. type: 11为模式矢量数据 =============
                data_type = 11  # 矢量网格数据
                f.write(np.int16(data_type).tobytes())

                # ============= 3. modelName =============
                model_name_bytes = model_name.upper().encode('ascii', 'ignore')[:20]
                if len(model_name_bytes) < 20:
                    model_name_bytes += b'\x00' * (20 - len(model_name_bytes))
                f.write(model_name_bytes)

                # ============= 4. element =============
                element_std = "10WIND"
                element_bytes = element_std.encode('ascii', 'ignore')[:50]
                if len(element_bytes) < 50:
                    element_bytes += b'\x00' * (50 - len(element_bytes))
                f.write(element_bytes)

                # ============= 5. description =============
                desc = description
                if not desc:
                    desc = "10m wind speed(m/s) and direction(deg)"
                if len(desc) > 30:
                    desc = desc[:30]
                try:
                    desc_bytes = desc.encode('gbk', 'ignore')[:30]
                except (UnicodeEncodeError, AttributeError):
                    desc_bytes = desc.encode('ascii', 'ignore')[:30]
                if len(desc_bytes) < 30:
                    desc_bytes += b'\x00' * (30 - len(desc_bytes))
                f.write(desc_bytes)

                # ============= 6. level =============
                f.write(np.float32(level).tobytes())

                # ============= 7-10. 起报日期和时间 =============
                f.write(np.int32(local_time.year).tobytes())
                f.write(np.int32(local_time.month).tobytes())
                f.write(np.int32(local_time.day).tobytes())
                f.write(np.int32(local_time.hour).tobytes())

                # ============= 11. timezone =============
                timezone = 8  # 北京时区
                f.write(np.int32(timezone).tobytes())

                # ============= 12. period =============
                f.write(np.int32(forecast_hour).tobytes())

                # ============= 13-15. 经度范围信息 =============
                f.write(np.float32(start_lon).tobytes())
                f.write(np.float32(end_lon).tobytes())
                f.write(np.float32(lon_res).tobytes())

                # ============= 16. longitudeGridNumber =============
                f.write(np.int32(n_lon).tobytes())

                # ============= 17-19. 纬度范围信息 =============
                f.write(np.float32(start_lat).tobytes())
                f.write(np.float32(end_lat).tobytes())
                f.write(np.float32(lat_res).tobytes())

                # ============= 20. latitudeGridNumber =============
                f.write(np.int32(n_lat).tobytes())

                # ============= 21-23. 等值线相关信息 =============
                f.write(np.float32(0.0).tobytes())  # isolineStartValue
                f.write(np.float32(0.0).tobytes())  # isolineEndValue
                f.write(np.float32(0.0).tobytes())  # isolineSpace

                # ============= 24. Extent =============
                f.write(b'\x00' * 100)

                # ============= 数据区 =============
                # 处理NaN值
                wspd_clean = wspd.copy()
                wdir_clean = wdir.copy()
                nan_mask_wspd = np.isnan(wspd_clean)
                nan_mask_wdir = np.isnan(wdir_clean)
                if np.any(nan_mask_wspd):
                    wspd_clean[nan_mask_wspd] = 9999.0
                if np.any(nan_mask_wdir):
                    wdir_clean[nan_mask_wdir] = 9999.0

                # 风向转换：气象风向 → MICAPS矢量角度
                micaps_wdir = (270.0 - wdir_clean) % 360.0
                micaps_wdir = micaps_wdir % 360.0

                # 展平数据
                wspd_flat = wspd_clean.ravel(order='C').astype(np.float32)
                wdir_flat = micaps_wdir.ravel(order='C').astype(np.float32)

                # 写入风速和风向
                f.write(wspd_flat.tobytes())
                f.write(wdir_flat.tobytes())

            return True

        except Exception as e:
            return False


# ================= 数据缓存管理类 =================
class DataCacheManager:
    """数据缓存管理器"""

    def __init__(self, cache_dir: str = '/tmp/ecmwf_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_temperature_cache_path(self, base_time: datetime) -> str:
        """获取温度数据缓存路径"""
        time_str = base_time.strftime("%Y%m%d%H")
        return os.path.join(self.cache_dir, f"temp_cache_{time_str}.pkl")

    def save_temperature_data(self, base_time: datetime, temp_data: Dict[str, np.ndarray]):
        """保存温度数据到缓存"""
        cache_path = self.get_temperature_cache_path(base_time)
        try:
            with open(cache_path, 'wb') as pickle:
                pickle.dump(temp_data, pickle)
            return True
        except Exception:
            return False

    def load_temperature_data(self, base_time: datetime) -> Optional[Dict[str, np.ndarray]]:
        """从缓存加载温度数据"""
        cache_path = self.get_temperature_cache_path(base_time)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as pickle:
                    return pickle.load()
            except Exception:
                return None
        return None


# ================= 数据处理类 =================
class ECDataProcessor:
    """ECMWF数据处理类 - 从NetCDF文件开始处理"""

    def __init__(self, logger: logging.Logger = None,
                 config: Dict = None,
                 save_micaps4: bool = False,
                 micaps4_output_dir: str = None,
                 cache_manager: DataCacheManager = None):
        """
        初始化处理器

        Parameters
        ----------
        logger : logging.Logger, optional
            日志记录器
        config : dict, optional
            配置字典
        save_micaps4 : bool
            是否保存MICAPS4格式
        micaps4_output_dir : str
            MICAPS4输出目录
        cache_manager : DataCacheManager, optional
            缓存管理器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.save_micaps4 = save_micaps4

        # 设置 MICAPS4 输出目录
        if micaps4_output_dir is not None:
            self.micaps4_output_dir = micaps4_output_dir
        elif config is not None and save_micaps4:
            self.micaps4_output_dir = config.get('micaps4_output_dir', None)
        else:
            self.micaps4_output_dir = None

        self.cache_manager = cache_manager or DataCacheManager()

        # 更新配置
        if config is not None:
            Config.REGION.update(config.get('REGION', {}))
            if 'RESOLUTION' in config:
                Config.RESOLUTION = config['RESOLUTION']

        # 生成目标网格
        self.lat_dst, self.lon_dst = Config.get_target_grid()
        self.n_lat_dst = len(self.lat_dst)
        self.n_lon_dst = len(self.lon_dst)

        if logger:
            logger.info(f"目标网格: {self.n_lat_dst}x{self.n_lon_dst}, 分辨率: {Config.RESOLUTION}度")

    def _process_scalar_data(self, element: str, nc_file: str,
                             base_time_utc: datetime, output_path: str) -> Tuple[
                                 bool, Dict[str, np.ndarray], List[datetime]]:
        """处理标量数据（从NetCDF文件读取）"""
        try:
            self.logger.info(f"处理标量要素: {element}")

            # 获取要素配置
            element_config = Config.ELEMENTS[element]

            # 1. 读取NetCDF数据
            start_time = time.time()
            with Dataset(nc_file, 'r') as ds:
                # 获取数据变量（排除坐标变量）
                data_vars = [v for v in ds.variables.keys()
                             if v not in ['time', 'latitude', 'longitude', 'lat', 'lon', 'level']]
                if not data_vars:
                    self.logger.error("NetCDF文件中未找到数据变量")
                    return False, None, None

                var_name = data_vars[0]
                self.logger.debug(f"使用数据变量: {var_name}")

                # 读取坐标
                if 'latitude' in ds.variables:
                    lat_var = ds.variables['latitude']
                    lon_var = ds.variables['longitude']
                elif 'lat' in ds.variables:
                    lat_var = ds.variables['lat']
                    lon_var = ds.variables['lon']
                else:
                    self.logger.error("NetCDF文件中未找到经纬度坐标变量")
                    return False, None, None

                # 读取经纬度
                lat_src = lat_var[:]
                lon_src = lon_var[:]

                # 确保纬度升序（从南到北）
                if lat_src[-1] < lat_src[0]:
                    self.logger.info("源纬度是降序，需要反转")
                    lat_src = lat_src[::-1]
                    # 注意：data_cube 在后面读取，稍后需要反转
                    # 标记需要反转纬度维度
                    need_reverse_lat = True
                else:
                    need_reverse_lat = False

                # 读取时间
                time_values = None
                steps = None
                if 'time' in ds.variables:
                    time_var = ds.variables['time']
                    time_values = time_var[:].flatten()

                    # 解析时间单位获取基准时间
                    time_base = base_time_utc  # 默认使用base_time
                    try:
                        time_units = time_var.units
                        if 'hours since' in time_units:
                            time_str = time_units.split('since')[1].strip()
                            time_formats = [
                                "%Y-%m-%d %H:%M:%S",
                                "%Y-%m-%d %H:%M",
                                "%Y-%m-%d",
                                "%Y %m %d %H"
                            ]
                            for fmt in time_formats:
                                try:
                                    time_base = datetime.strptime(time_str, fmt)
                                    break
                                except ValueError:
                                    continue
                            else:
                                self.logger.warning(f"无法解析时间基准: {time_str}")
                    except Exception as e:
                        self.logger.warning(f"时间单位解析失败: {str(e)}")

                    # 转换为预报时次
                    steps = []
                    for t in time_values:
                        try:
                            forecast_time = time_base + timedelta(hours=float(t))
                            forecast_hour = int((forecast_time - base_time_utc).total_seconds() / 3600)
                            steps.append(forecast_hour)
                        except:
                            pass

                    # 使用set去重后排序
                    steps = sorted(set(steps))
                    if steps:
                        # 按时间排序（处理CDO可能产生的乱序时间）
                        # 创建(时间, 索引, 数据)的列表
                        # 先读取所有数据
                        data_var = ds.variables[var_name]
                        all_data = data_var[:]  # shape: (time, lat, lon)

                        # 处理masked array：如果数据有mask，转换为普通numpy array并用NaN填充
                        if hasattr(all_data, 'mask'):
                            all_data = all_data.filled(np.nan)
                            self.logger.debug("数据包含masked array，已转换为普通array并用NaN填充")

                        # 创建时间-数据对
                        time_data_pairs = list(zip(steps, range(len(steps))))
                        # 按时间排序
                        time_data_pairs.sort(key=lambda x: x[0])
                        # 重新提取排序后的steps和数据
                        steps_sorted = [item[0] for item in time_data_pairs]
                        indices_sorted = [item[1] for item in time_data_pairs]
                        # 按排序后的索引重新读取数据
                        data_cube = all_data[indices_sorted, :, :]
                        # 如果源纬度原本是降序，需要反转数据中的纬度维度
                        if need_reverse_lat:
                            self.logger.info("数据中的纬度维度需要反转以匹配升序坐标")
                            data_cube = data_cube[:, ::-1, :]
                        self.logger.debug(f"时间排序后，steps前10个: {steps_sorted[:10]}")
                    else:
                        self.logger.error("没有有效的预报时次")
                        return False, None, None
                if steps is None:
                    self.logger.error("无法读取时间信息")
                    return False, None, None

            read_time = time.time() - start_time
            self.logger.info(f"NetCDF数据读取完成: {read_time:.1f}秒, 时次: {len(steps)}")

            # 2. 空间插值
            start_time = time.time()

            # 准备目标网格点
            lon2d, lat2d = np.meshgrid(self.lon_dst, self.lat_dst)
            points = np.column_stack([lat2d.ravel(), lon2d.ravel()])

            # 批量插值
            data_interp = np.empty((len(steps), self.n_lat_dst, self.n_lon_dst), dtype=np.float32)
            for i in range(len(steps)):
                # 为每个时次创建新的插值器
                interp_func = RegularGridInterpolator(
                    (lat_src, lon_src),
                    data_cube[i],
                    bounds_error=False,
                    fill_value=np.nan
                )
                data_interp[i] = interp_func(points).reshape(self.n_lat_dst, self.n_lon_dst)

            interp_time = time.time() - start_time
            self.logger.info(f"空间插值完成: {interp_time:.1f}秒")

            # 3. 单位转换（如果配置了转换）
            if element_config.get("conversion") == "K_to_C":
                self.logger.info(f"单位转换: 开尔文(K) → 摄氏度(°C)")
                data_interp = MeteorologicalCalculator.kelvin_to_celsius(data_interp)

            # 降水特殊处理：累计降水 → 时段降水 → 小时降水
            is_precip = (element == "PRE" or element_config.get("grib_codes", {}).get("value") == "tp")
            if is_precip:
                self.logger.info("检测到降水数据，进行累计降水到小时降水的转换")

            # 4. 时间插值（分段处理）
            start_time = time.time()

            # 根据时效分两段
            steps_3h = [step for step in steps if step <= 72]
            steps_6h = [step for step in steps if step > 72 and step <= 240]

            # GUST (10fg3) 特殊处理：只有3小时间隔，无6小时间隔
            is_gust = (var_name == 'gust' or element_config.get("grib_codes", {}).get("value") == "10fg3")

            # MN2T6/MX2T6（过去6小时最低/最高气温）特殊处理：起始时间点是6小时
            is_mn2t6 = (var_name == 'mn2t6')
            is_mx2t6 = (var_name == 'mx2t6')

            # 分段处理
            data_1h_segments = []
            hour_segments = []

            # 第一段插值起始点：根据要素类型确定
            if steps_3h:
                indices_3h = [i for i, step in enumerate(steps) if step in steps_3h]
                data_3h = data_interp[indices_3h]
                hours_3h = steps_3h

                # MN2T6和MX2T6从6小时开始，其他从0小时开始
                if is_mn2t6 or is_mx2t6:
                    # 过去6小时要素：从6开始插值到72
                    hours_new_3h = np.arange(6, 73, 1.0)
                else:
                    # TEM等常规要素：从0开始插值到72
                    hours_new_3h = np.arange(0, 73, 1.0)

                # 降水特殊处理：累计降水转小时降水
                if is_precip:
                    # 1. 计算时段降水（后一时刻累积 - 前一时刻累积）
                    self.logger.info("  计算时段降水（累计差值）...")
                    n_intervals = len(hours_3h)
                    interval_data = np.empty((n_intervals, data_3h.shape[1], data_3h.shape[2]), dtype=np.float32)
                    interval_hours = []
                    for i in range(n_intervals):
                        interval_data[i] = data_3h[i+1] - data_3h[i]
                        interval_hours.append(hours_3h[i+1])  # 时段**结束时间**，不是开始时间
                        # 检查负值（由于数值误差可能出现微小负值）
                        negative_mask = interval_data[i] < -0.001
                        if np.any(negative_mask):
                            interval_data[i][negative_mask] = 0.0

                    # 2. 降尺度到1小时（平均分配到时段）
                    self.logger.info(f"  降尺度到1小时降水...")
                    total_hours_3h = len(hours_new_3h)
                    data_1h_3h = np.zeros((total_hours_3h, data_3h.shape[1], data_3h.shape[2]), dtype=np.float32)

                    for i, (start_hour, precip) in enumerate(zip(interval_hours, interval_data)):
                        interval_len = interval_hours[i+1] - interval_hours[i] if i < len(interval_hours)-1 else 1

                        # 计算在输出数组中的索引
                        if i < len(interval_hours) - 1:
                            # 普通时段
                            local_start = max(start_hour - hours_new_3h[0], 0)
                            local_end = min(local_start + interval_len, total_hours_3h)

                            if interval_len > 0 and local_start < local_end:
                                # 平均分配到每个小时（时段降水/间隔）
                                hourly_amount = precip / float(interval_len)
                                for hour_offset in range(local_end - local_start):
                                    if local_start + hour_offset < total_hours_3h:
                                        data_1h_3h[local_start + hour_offset] = hourly_amount
                        else:
                            # 最后一个时段
                            local_start = max(start_hour - hours_new_3h[0], 0)
                            if local_start < total_hours_3h:
                                data_1h_3h[local_start] = precip

                    self.logger.info(f"  0-72小时降水: {len(hours_3h)}个原始时次 → {total_hours_3h}个1小时间隔时次（累计转时段转小时）")
                else:
                    # 常规要素：线性插值
                    f_data_3h = interp1d(hours_3h, data_3h, axis=0, kind='linear',
                                        bounds_error=False, fill_value=np.nan)
                    data_1h_3h = f_data_3h(hours_new_3h)
                    self.logger.info(f"  0-72小时: {len(hours_3h)}个原始时次 → {len(hours_new_3h)}个1小时间隔时次")

                data_1h_segments.append(data_1h_3h)
                hour_segments.append(hours_new_3h)
            else:
                self.logger.warning("  0-72小时段无有效数据")

            # 第二段插值（73-240小时）
            if steps_6h:
                indices_6h = [i for i, step in enumerate(steps) if step in steps_6h]
                data_6h = data_interp[indices_6h]
                hours_6h = steps_6h

                # 计算第二段的起始小时（第一段结束+1）
                if hour_segments and len(hour_segments[-1]) > 0:
                    last_hour_3h = hour_segments[-1][-1]
                    start_hour_6h = last_hour_3h + 1
                else:
                    start_hour_6h = 73  # 如果没有第一段数据，从73开始

                # 计算第二段的结束小时
                end_hour_6h = start_hour_6h + len(hours_6h) * 6 - 1

                # MN2T6和MX2T6特殊处理：6小时间隔，不需要插值
                if is_mn2t6 or is_mx2t6:
                    # 直接取每隔6小时的数据
                    hours_new_6h = []
                    data_1h_6h = []

                    # 需要计算的是过去6小时的极值，所以需要累积处理
                    # 这里简化处理：直接从6小时间隔数据中提取
                    # 实际应用中可能需要更复杂的逻辑

                    for i, hour in enumerate(hours_6h):
                        # 过去6小时极值在6小时间隔点上
                        if (hour - 6) >= 0:  # 确保有足够的过去数据
                            hours_new_6h.append(hour)
                            # 这里简化处理，实际应该根据具体需求计算
                            data_1h_6h.append(data_6h[i])

                    if data_1h_6h:
                        data_1h_6h = np.stack(data_1h_6h, axis=0)
                    else:
                        data_1h_6h = np.zeros((len(hours_new_6h), data_6h.shape[1], data_6h.shape[2]), dtype=np.float32)

                    self.logger.info(f"  {start_hour_6h}-{end_hour_6h}小时极值: {len(hours_6h)}个原始时次 → {len(hours_new_6h)}个1小时间隔时次（6小时间隔）")
                else:
                    # 常规要素：从6小时间隔插值到1小时
                    hours_new_6h = np.arange(start_hour_6h, end_hour_6h + 1, 1.0)

                    # 降水特殊处理：累计降水转小时降水
                    if is_precip:
                        # 1. 计算时段降水（后一时刻累积 - 前一时刻累积）
                        self.logger.info("  计算时段降水（累计差值）...")
                        n_intervals = len(hours_6h)
                        interval_data = np.empty((n_intervals, data_6h.shape[1], data_6h.shape[2]), dtype=np.float32)
                        interval_hours = []

                        for i in range(n_intervals):
                            if i < len(hours_6h) - 1:
                                interval_data[i] = data_6h[i+1] - data_6h[i]
                                interval_hours.append(hours_6h[i+1])  # 时段**结束时间**
                                # 检查负值
                                negative_mask = interval_data[i] < -0.001
                                if np.any(negative_mask):
                                    interval_data[i][negative_mask] = 0.0

                        # 2. 降尺度到1小时（平均分配到时段）
                        self.logger.info(f"  降尺度到1小时降水...")
                        total_hours_6h = len(hours_new_6h)
                        data_1h_6h = np.zeros((total_hours_6h, data_6h.shape[1], data_6h.shape[2]), dtype=np.float32)

                        # 插入第一段和第二段之间的间隔（72-78小时）
                        # 第一段最后一个时段是69-72小时，第二段第一个时段是78-84小时
                        # 72-78这6个小时降水：(tp[78h] - tp[72h])，平均分配到6个小时
                        # 插入到72和78之间
                        hours_72_78 = [72, 73, 74, 75, 76, 77]  # 6个小时
                        if len(hours_6h) > 0 and 72 in hours_6h and 78 in hours_6h:
                            idx_72 = hours_6h.index(72)
                            idx_78 = hours_6h.index(78)
                            precip_72_78 = data_6h[idx_78] - data_6h[idx_72]

                            # 将72-78小时的降水均匀分配到6个小时
                            for i, hour in enumerate(hours_72_78):
                                if start_hour_6h + i < total_hours_6h:
                                    data_1h_6h[start_hour_6h + i] = precip_72_78 / 6.0

                        # 处理其他6小时时段
                        for i, (start_hour, precip) in enumerate(zip(interval_hours, interval_data)):
                            if i < len(interval_hours) - 1:
                                interval_len = interval_hours[i+1] - interval_hours[i]
                                local_start = max(start_hour - start_hour_6h, 0)
                                local_end = min(local_start + interval_len, total_hours_6h)

                                if interval_len > 0 and local_start < local_end:
                                    hourly_amount = precip / float(interval_len)
                                    for hour_offset in range(local_end - local_start):
                                        if local_start + hour_offset < total_hours_6h:
                                            data_1h_6h[local_start + hour_offset] = hourly_amount

                        self.logger.info(f"  {start_hour_6h}-{end_hour_6h}小时降水: {len(hours_6h)}个原始时次 → {len(hours_new_6h)}个1小时间隔时次（累计转时段转小时）")
                    else:
                        # 常规要素：线性插值
                        f_data_6h = interp1d(hours_6h, data_6h, axis=0, kind='linear',
                                            bounds_error=False, fill_value=np.nan)
                        data_1h_6h = f_data_6h(hours_new_6h)
                        self.logger.info(f"  {start_hour_6h}-{end_hour_6h}小时: {len(hours_6h)}个原始时次 → {len(hours_new_6h)}个1小时间隔时次")

                data_1h_segments.append(data_1h_6h)
                hour_segments.append(hours_new_6h)
            elif is_gust or not steps_6h or max(steps) <= 72:
                self.logger.warning(f"  72-240小时段无有效数据")

            # 合并所有分段
            if data_1h_segments:
                data_1h = np.concatenate(data_1h_segments, axis=0)
                hours_new = np.concatenate(hour_segments, axis=0)

                # 生成北京时时间序列
                times_bjt = [base_time_utc + timedelta(hours=float(h)) + Config.TIMEZONE_SHIFT
                             for h in hours_new]

                # MN2T6 和 MX2T6：前面缺少的时间点用 NaN 填充（像 GUST 一样）
                if is_mn2t6 or is_mx2t6:
                    # 找到数据的起始时间点
                    start_hour = int(hours_new[0])
                    if start_hour > 0:
                        # 需要在前面添加 NaN
                        n_nan_hours = start_hour
                        nan_data = np.full((n_nan_hours, data_1h.shape[1], data_1h.shape[2]), np.nan, dtype=np.float32)
                        # 合并：NaN 前置 + 数据
                        data_1h = np.concatenate([nan_data, data_1h], axis=0)
                        # 重新生成时间序列（从0开始）
                        hours_new = np.arange(0, len(data_1h), 1.0)
                        times_bjt = [base_time_utc + timedelta(hours=float(h)) + Config.TIMEZONE_SHIFT
                                     for h in hours_new]
                        self.logger.info(f"  已为 {var_name} 填充前面 {n_nan_hours} 个小时为 NaN")

                time_interp_time = time.time() - start_time
                self.logger.info(f"时间插值完成: {time_interp_time:.1f}秒")

                # 保存数据
                success, saved_vars = self._save_processed_data(
                    element, data_1h, times_bjt, hours_new, output_path, element_config
                )

                if success:
                    self.logger.info(f"要素 {element} 处理完成，保存了 {len(saved_vars)} 个变量")
                    return True, saved_vars, times_bjt
                else:
                    self.logger.error(f"要素 {element} 数据保存失败")
                    return False, None, None
            else:
                self.logger.error("没有有效的时间段数据")
                return False, None, None

        except Exception as e:
            self.logger.error(f"处理标量数据时发生错误: {str(e)}")
            return False, None, None

    def _save_processed_data(self, element: str, data_1h: np.ndarray, times_bjt: List[datetime],
                            hours_new: np.ndarray, output_path: str,
                            element_config: Dict) -> Tuple[bool, Dict[str, np.ndarray]]:
        """
        保存处理后的数据到NetCDF文件

        Parameters
        ----------
        element : str
            要素名称
        data_1h : np.ndarray
            1小时分辨率的数据 (time, lat, lon)
        times_bjt : List[datetime]
            北京时时间列表
        hours_new : np.ndarray
            预报时效（小时）
        output_path : str
            输出文件路径
        element_config : Dict
            要素配置

        Returns
        -------
        tuple
            (是否成功, 保存的变量字典)
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 创建NetCDF文件
            with Dataset(output_path, 'w', format='NETCDF4') as nc_out:
                # 定义维度
                nc_out.createDimension('time', len(hours_new))
                nc_out.createDimension('lat', self.n_lat_dst)
                nc_out.createDimension('lon', self.n_lon_dst)

                # 创建坐标变量
                # 时间
                time_var = nc_out.createVariable('time', 'f8', ('time',))
                time_var.units = f'hours since {times_bjt[0].strftime("%Y-%m-%d %H:%M:%S")}'
                time_var.calendar = 'gregorian'
                time_var[:] = hours_new

                # 纬度
                lat_var = nc_out.createVariable('lat', 'f4', ('lat',))
                lat_var.units = 'degrees_north'
                lat_var.long_name = 'latitude'
                lat_var[:] = self.lat_dst

                # 经度
                lon_var = nc_out.createVariable('lon', 'f4', ('lon',))
                lon_var.units = 'degrees_east'
                lon_var.long_name = 'longitude'
                lon_var[:] = self.lon_dst

                # 保存输出变量
                output_vars = element_config.get('output_vars', [])
                units = element_config.get('units', {})
                saved_vars = {}

                for var_name in output_vars:
                    # 获取单位
                    unit = units.get(var_name, '')

                    # 创建变量
                    nc_var = nc_out.createVariable(var_name, 'f4', ('time', 'lat', 'lon'), fill_value=np.nan)
                    nc_var.units = unit
                    nc_var.long_name = f"{element} {var_name}"

                    # 写入数据
                    nc_var[:] = data_1h

                    saved_vars[var_name] = data_1h

                    # 如果需要保存MICAPS4格式
                    if self.save_micaps4 and self.micaps4_output_dir:
                        self._save_micaps4_for_element(
                            element, var_name, data_1h, times_bjt, hours_new,
                            unit, element_config
                        )

                # 添加全局属性
                nc_out.title = f"ECMWF {element} data processed to 0.01 degree resolution"
                nc_out.source = "ECMWF"
                nc_out.institution = "ECMWF"
                nc_out.Conventions = "CF-1.6"
                nc_out.history = f"Created {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            self.logger.info(f"数据已保存到: {output_path}")
            return True, saved_vars

        except Exception as e:
            self.logger.error(f"保存数据时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, {}

    def _save_micaps4_for_element(self, element: str, var_name: str,
                                 data_1h: np.ndarray, times_bjt: List[datetime],
                                 hours_new: np.ndarray, unit: str,
                                 element_config: Dict):
        """
        为指定要素保存MICAPS4格式文件

        Parameters
        ----------
        element : str
            要素名称
        var_name : str
            变量名称
        data_1h : np.ndarray
            1小时分辨率的数据 (time, lat, lon)
        times_bjt : List[datetime]
            北京时时间列表
        hours_new : np.ndarray
            预报时效（小时）
        unit : str
            单位
        element_config : Dict
            要素配置
        """
        try:
            if not self.micaps4_output_dir:
                return

            # 为每个时间步创建MICAPS4文件
            base_time_utc = times_bjt[0] - Config.TIMEZONE_SHIFT

            for i, (time_bjt, hour) in enumerate(zip(times_bjt, hours_new)):
                # 提取该时间步的数据
                data_2d = data_1h[i]

                # 创建输出路径
                # 格式: /micaps4_dir/ELEMENT/YYMMDDHH.TTT
                element_dir = os.path.join(self.micaps4_output_dir, element)
                os.makedirs(element_dir, exist_ok=True)

                micaps_filename = MICAPS4Writer.create_micaps4_filename(
                    base_time_utc, int(hour), element, "ECMWF", Config.TIMEZONE_SHIFT
                )
                micaps_output_path = os.path.join(element_dir, micaps_filename)

                # 检查是否跳过已存在的文件
                if os.path.exists(micaps_output_path):
                    continue

                # 写入MICAPS4文件
                success = MICAPS4Writer.write_micaps4_scalar_file(
                    data_2d, self.lat_dst, self.lon_dst,
                    base_time_utc, int(hour),
                    micaps_output_path, element, "ECMWF",
                    level=1000.0,
                    description=f"{element} {unit}",
                    timezone_shift=Config.TIMEZONE_SHIFT
                )

                if not success:
                    self.logger.warning(f"MICAPS4文件保存失败: {micaps_output_path}")

        except Exception as e:
            self.logger.error(f"保存MICAPS4文件时发生错误: {str(e)}")


# ================= 主程序入口 =================
def main():
    """主程序入口"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='ECMWF多要素数据处理工具 - 从NetCDF文件开始处理不包含CDO转换步骤，专注于数据处理',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""使用示例：
批量处理 nc_output 目录下的所有NC文件：
  python3 %(prog)s --input-dir /path/to/nc_output --output-dir /path/to/output --verbose

只处理某个要素：
  python3 %(prog)s --input-dir /path/to/nc_output --element TEM --output-dir /path/to/output

单个文件处理：
  python3 %(prog)s --element TEM --nc-file /path/to/file.nc --base-time 2026022612

不跳过已存在文件（强制重新处理）：
  python3 %(prog)s ... --no-skip-existing
"""
    )

    # 必选参数
    group_required = parser.add_mutually_exclusive_group(required=True)
    group_required.add_argument('--input-dir', type=str,
                               help='输入目录：包含多个NC文件的目录')
    group_required.add_argument('--nc-file', type=str,
                               help='单个NC文件路径')

    # 其他参数
    parser.add_argument('--element', type=str, default=None,
                       help='只处理指定的要素（如：TEM, WIND_10M, PRE等）')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='输出目录，默认为./output')
    parser.add_argument('--base-time', type=str,
                       help='基准时间（格式：YYYYMMDDHH），从文件名解析时不需要')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='显示详细处理信息')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别，默认INFO')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='跳过已存在的输出文件（默认启用）')
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='不跳过已存在的输出文件，强制重新处理')
    parser.add_argument('--save-micaps4', action='store_true',
                       help='同时保存MICAPS4格式文件')
    parser.add_argument('--micaps4-dir', type=str, default=None,
                       help='MICAPS4输出目录，默认为output_dir/micaps4')

    # 解析参数
    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # 处理冲突参数
    if args.no_skip_existing:
        args.skip_existing = False
    elif args.skip_existing:
        logger.info("将跳过已存在的输出文件（使用 --no-skip-existing 强制重新处理）")

    # 解析基准时间
    base_time_utc = None
    if args.base_time:
        try:
            base_time_utc = datetime.strptime(args.base_time, "%Y%m%d%H")
        except ValueError:
            logger.error(f"基准时间格式错误: {args.base_time}，请使用YYYYMMDDHH格式")
            return 1

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_micaps4:
        micaps4_dir = args.micaps4_dir or os.path.join(args.output_dir, 'micaps4')
        os.makedirs(micaps4_dir, exist_ok=True)

    # 创建处理器
    config = {
        'micaps4_output_dir': micaps4_dir if args.save_micaps4 else None
    }
    processor = ECDataProcessor(
        logger=logger,
        config=config,
        save_micaps4=args.save_micaps4,
        micaps4_output_dir=micaps4_dir if args.save_micaps4 else None
    )

    # 获取要处理的文件列表
    nc_files = []
    if args.input_dir:
        # 扫描目录下的所有.nc文件
        for root, dirs, files in os.walk(args.input_dir):
            for file in files:
                if file.endswith('.nc'):
                    nc_files.append(os.path.join(root, file))
    elif args.nc_file:
        # 处理单个文件
        if os.path.exists(args.nc_file):
            nc_files = [args.nc_file]
        else:
            logger.error(f"文件不存在: {args.nc_file}")
            return 1

    if not nc_files:
        logger.error("未找到任何NC文件")
        return 1

    logger.info(f"找到 {len(nc_files)} 个NC文件")

    # 从文件名解析基准时间（如果未提供）
    if base_time_utc is None:
        # 使用第一个文件的基准时间
        first_file = nc_files[0]
        base_time_utc = Config.parse_time_from_filename(first_file)
        if base_time_utc:
            logger.info(f"从文件名解析得到基准时间: {base_time_utc}")
        else:
            logger.error("无法从文件名解析基准时间，请使用 --base-time 参数指定")
            return 1

    # 处理每个文件
    success_count = 0
    fail_count = 0
    processed_elements = set()

    for nc_file in nc_files:
        try:
            basename = os.path.basename(nc_file)
            logger.info(f"\n正在处理文件: {basename}")

            # 确定要处理的要素
            if args.element:
                elements = [args.element]
            else:
                # 尝试从文件名推断要素
                elements = []
                for elem_name in Config.ELEMENTS.keys():
                    if elem_name.lower() in basename.lower():
                        elements.append(elem_name)
                if not elements:
                    # 如果无法推断，处理所有要素
                    elements = list(Config.ELEMENTS.keys())

            logger.info(f"将处理要素: {', '.join(elements)}")

            # 处理每个要素
            for element in elements:
                # 检查是否跳过
                output_filename = Config.OUTPUT_FILENAME_FORMAT.format(
                    element=element,
                    time_str=Config.utc_to_bjt_str(base_time_utc)
                )
                output_path = os.path.join(args.output_dir, output_filename)

                if args.skip_existing and os.path.exists(output_path):
                    logger.info(f"输出文件已存在，跳过: {output_filename}")
                    continue

                # 处理数据
                success, result_vars, times = processor._process_scalar_data(
                    element, nc_file, base_time_utc, output_path
                )

                if success:
                    logger.info(f"✓ 成功处理: {element} -> {output_filename}")
                    success_count += 1
                    processed_elements.add(element)
                else:
                    logger.error(f"✗ 处理失败: {element}")
                    fail_count += 1

        except Exception as e:
            logger.error(f"处理文件 {basename} 时发生错误: {str(e)}")
            fail_count += 1

    # 总结
    logger.info(f"\n处理完成！")
    logger.info(f"成功: {success_count} 个要素")
    logger.info(f"失败: {fail_count} 个要素")
    if processed_elements:
        logger.info(f"处理的要素: {', '.join(sorted(processed_elements))}")

    return 0 if fail_count == 0 else 1


# ================= 程序入口 =================
if __name__ == '__main__':
    sys.exit(main())