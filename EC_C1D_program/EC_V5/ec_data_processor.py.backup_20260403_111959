#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECMWF多要素数据处理工具 - 从NetCDF文件开始处理
不包含CDO转换步骤，专注于数据处理


 # 批量处理 nc_output 目录下的所有NC文件                                                                                                                                                
  python3 /home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/ec_data_processor.py \                                                                                                     
      --input-dir /home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/nc_output \                                                                                                        
      --output-dir /home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/1KM1H \                                                                                                           
      --verbose
                                                                                                                                                                                         
  其他常用选项：                                                                                                                                                                         
                                                                                                                                                                                         
  # 只处理某个要素                                                                                                                                                                       
  python3 /home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/ec_data_processor.py \                                                                                                     
      --input-dir /home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/nc_output \
      --element TEM \                                                                                                                                                                    
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
import gc


# ================= 配置类 =================
class Config:
    """集中管理所有配置参数"""
    # 区域范围
    REGION = {
        "lon_w": 95.0,
        "lon_e": 127.0,
        "lat_s": 33.0,
        "lat_n": 48.0
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
            "conversion": "K_to_C",  # 添加转换标志
            "nc_mapping": {
                "input_pattern": "ECMFC1D_TEM",
                "input_var": "2t",
                "output_filename": "tem"
            }
        },
        # 风场
        "WIND": {
            "description": "10m wind",
            "grib_codes": {"u": "10u", "v": "10v"},
            "requires_uv": True,
            "output_vars": ["u", "v"],
            "units": {"u": "m s-1", "v": "m s-1"},
            "nc_mapping": {
                "input_pattern": "ECMFC1D_10",
                "input_vars": {"u": "10u", "v": "10v"},
                "output_filename": "wind_10m"
            }
        },
        # 阵风
        "GUST": {
            "description": "10m wind gust",
            "grib_codes": {"value": "10fg3"},
            "requires_uv": False,
            "output_vars": ["gust"],
            "units": {"gust": "m s-1"},
            "conversion": None,
            "nc_mapping": {
                "input_pattern": "ECMFC1D_10FG3",
                "input_var": "10fg3",
                "output_filename": "gust"
            }
        },
        # 气压
        "PRS": {
            "description": "surface pressure",
            "grib_codes": {"value": "msl"},
            "requires_uv": False,
            "output_vars": ["prs"],
            "units": {"prs": "Pa"},
            "conversion": None,
            "nc_mapping": {
                "input_pattern": "ECMFC1D_PRS",
                "input_var": "sp",
                "output_filename": "prs"
            }
        },
        # 露点温度
        "DPT": {
            "description": "2m dewpoint temperature",
            "grib_codes": {"value": "2d"},
            "requires_uv": False,
            "output_vars": ["dpt"],
            "units": {"dpt": "°C"},  # 修改为摄氏度
            "conversion": "K_to_C",  # 添加转换标志
            "nc_mapping": {
                "input_pattern": "ECMFC1D_DPT",
                "input_var": "2d",
                "output_filename": "dpt"
            }
        },
        # 相对湿度
        "RH": {
            "description": "2m relative humidity",
            "grib_codes": {"temp": "TEM", "prs": "PRS", "dpt": "DPT"},
            "requires_uv": False,
            "requires_calc": True,
            "output_vars": ["rh"],
            "units": {"rh": "%"},
            "nc_mapping": {
                "input_pattern": "RH",  # RH是计算出来的，不直接从文件读取
                "input_var": "rh",
                "output_filename": "rh"
            }
        },
        # 最低温度
        "MN2T6": {
            "description": "minimum temperature in 2m height in past 6 hours",
            "grib_codes": {"value": "mn2t6"},
            "requires_uv": False,
            "output_vars": ["mn2t6"],
            "units": {"mn2t6": "°C"},
            "conversion": "K_to_C",
            "nc_mapping": {
                "input_pattern": "ECMFC1D_MN2T6",
                "input_var": "mn2t6",
                "output_filename": "mn2t6"
            }
        },
        # 最高温度
        "MX2T6": {
            "description": "maximum temperature in 2m height in past 6 hours",
            "grib_codes": {"value": "mx2t6"},
            "requires_uv": False,
            "output_vars": ["mx2t6"],
            "units": {"mx2t6": "°C"},
            "conversion": "K_to_C",
            "nc_mapping": {
                "input_pattern": "ECMFC1D_MX2T6",
                "input_var": "mx2t6",
                "output_filename": "mx2t6"
            }
        },
        # 降水（PRE文件）
        "PRE": {
            "description": "total precipitation",
            "grib_codes": {"value": "tp"},
            "requires_uv": False,
            "output_vars": ["pre"],
            "units": {"pre": "mm"},
            "conversion": None,
            "nc_mapping": {
                "input_pattern": "ECMFC1D_PRE",  # 需要有PRE文件
                "input_var": "tp",
                "output_filename": "pre"
            }
        },
        # 降水（TPE文件）
        "TPE": {
            "description": "total precipitation (TPE)",
            "grib_codes": {"value": "tp"},
            "requires_uv": False,
            "output_vars": ["tpe"],
            "units": {"tpe": "mm"},
            "conversion": None,
            "nc_mapping": {
                "input_pattern": "ECMFC1D_TPE",
                "input_var": "tp",
                "output_filename": "tpe"
            }
        },
        # 能见度
        "VIS": {
            "description": "visibility",
            "grib_codes": {"value": "vis"},
            "requires_uv": False,
            "output_vars": ["vis"],
            "units": {"vis": "m"},
            "conversion": None,
            "nc_mapping": {
                "input_pattern": "ECMFC1D_VIS",
                "input_var": "vis",
                "output_filename": "vis"
            }
        },
        # 总云量
        "TCC": {
            "description": "total cloud cover",
            "grib_codes": {"value": "tcc"},
            "requires_uv": False,
            "output_vars": ["tcc"],
            "units": {"tcc": "%"},
            "conversion": None,
            "nc_mapping": {
                "input_pattern": "ECMFC1D_TCC",
                "input_var": "tcc",
                "output_filename": "tcc"
            }
        },
        # 100m风场
        "WIND100": {
            "description": "100m wind",
            "grib_codes": {"u": "100u", "v": "100v"},
            "requires_uv": True,
            "output_vars": ["u", "v"],
            "units": {"u": "m s-1", "v": "m s-1"},
            "nc_mapping": {
                "input_pattern": "ECMFC1D_100",
                "input_vars": {"u": "100u", "v": "100v"},
                "output_filename": "wind_100m"
            }
        },
        # 60m风场（从100m风场计算）
        "WIND60": {
            "description": "60m wind (calculated from 100m)",
            "grib_codes": {"u": "100u", "v": "100v"},  # 基于100m数据计算
            "requires_uv": True,
            "requires_100m": True,  # 需要100m数据
            "output_vars": ["u", "v"],
            "units": {"u": "m s-1", "v": "m s-1"},
            "nc_mapping": {
                "input_pattern": "ECMFC1D_100",  # 基于100m数据计算
                "input_vars": {"u": "100u", "v": "100v"},
                "output_filename": "wind_60m"
            }
        }
    }

    # 风廓线参数（用于60m风场计算）
    WIND_PROFILE_PARAMS = {
        "power_law_exponent": 0.143,  # 幂律指数，适用于中性大气
        "reference_height": 100.0,  # 参考高度（米）
        "target_height": 60.0,  # 目标高度（米）
        "log_law_roughness": 0.03  # 对数律粗糙度（米），适用于平坦地形
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

    @classmethod
    def get_element_from_filename(cls, filename: str) -> Optional[str]:
        """从文件名解析要素类型"""
        basename = os.path.basename(filename)

        # 根据文件名模式匹配要素，优先使用精确匹配
        for element, config in cls.ELEMENTS.items():
            if "nc_mapping" in config:
                pattern = config["nc_mapping"]["input_pattern"]
                # 使用更精确的匹配方式
                if pattern in ["ECMFC1D_10", "ECMFC1D_100"]:
                    # 对于这些风场要素，需要精确匹配
                    if pattern == "ECMFC1D_10" and "10U" in basename or "10V" in basename:
                        return element
                    elif pattern == "ECMFC1D_100" and ("100U" in basename or "100V" in basename):
                        return element
                else:
                    # 其他要素使用包含匹配
                    if pattern in basename:
                        return element

        return None


# ================= 气象计算工具类 =================
class MeteorologicalCalculator:
    """气象要素计算工具类"""

    @staticmethod
    def kelvin_to_celsius(kelvin_data: np.ndarray) -> np.ndarray:
        """开尔文转摄氏度"""
        return kelvin_data - 273.15

    @staticmethod
    def saturation_vapor_pressure(temp_c: np.ndarray) -> np.ndarray:
        """
        使用Magnus-Tetens公式计算饱和水汽压

        Parameters
        ----------
        temp_c : np.ndarray
            温度（摄氏度）

        Returns
        -------
        np.ndarray
            饱和水汽压（hPa）

        Notes
        -----
        Magnus-Tetens公式: e_s = 6.112 * exp(17.67 * T / (T + 243.5))
        其中T是摄氏温度，e_s是饱和水汽压（hPa）
        """
        return 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))

    @staticmethod
    def calculate_relative_humidity_from_temp_and_dewpoint(temp_c: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
        """
        从温度和露点温度计算相对湿度（精确版）

        Parameters
        ----------
        temp_c : np.ndarray
            温度（摄氏度）
        dewpoint_c : np.ndarray
            露点温度（摄氏度）

        Returns
        -------
        np.ndarray
            相对湿度（%）

        Notes
        -----
        计算方法：
        1. 使用Magnus-Tetens公式计算温度对应的饱和水汽压
        2. 使用Magnus-Tetens公式计算露点温度对应的实际水汽压
        3. 相对湿度 = 实际水汽压 / 饱和水汽压 * 100%
        """
        # 计算饱和水汽压
        e_s = MeteorologicalCalculator.saturation_vapor_pressure(temp_c)

        # 计算实际水汽压（从露点温度）
        e = MeteorologicalCalculator.saturation_vapor_pressure(dewpoint_c)

        # 计算相对湿度
        rh = (e / e_s) * 100.0

        # 限制在0-100%范围内
        rh = np.clip(rh, 0.0, 100.0)

        # 处理NaN值
        rh = np.where(np.isnan(temp_c) | np.isnan(dewpoint_c), np.nan, rh)

        return rh

    @staticmethod
    def calculate_relative_humidity_from_temperature(temp_c: np.ndarray) -> np.ndarray:
        """
        直接从温度计算相对湿度（简化版）
        注意：这是一个简化算法，实际应用中可能需要更精确的公式
        """
        # 简化算法：基于温度估算相对湿度
        # 这里使用一个简单的经验公式，实际应用中应该使用完整的温度-露点温度公式

        # 创建一个模拟的相对湿度场
        # 在实际应用中，这里应该使用完整的温度、气压、露点温度数据
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
        u = -wspd * np.sin(wdir_rad)  # 注意：气象风向定义中，U正方向为西风
        v = -wspd * np.cos(wdir_rad)  # V正方向为南风

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
        params = Config.WIND_PROFILE_PARAMS
        z_ref = params["reference_height"]
        z_target = params["target_height"]

        # 计算风速
        wspd_100m, wdir_100m = MeteorologicalCalculator.calculate_wind_speed_direction(u_100m, v_100m)

        if method == 'power_law':
            # 幂律方法
            alpha = params["power_law_exponent"]
            wspd_60m = wspd_100m * (z_target / z_ref) ** alpha

        elif method == 'log_law':
            # 对数律方法
            z0 = params["log_law_roughness"]
            wspd_60m = wspd_100m * (np.log(z_target / z0) / np.log(z_ref / z0))
        else:
            raise ValueError(f"未知的计算方法: {method}")

        # 风向通常随高度变化不大，使用相同的风向
        wdir_60m = wdir_100m.copy()

        # 将60米风速风向转换回U/V分量
        u_60m, v_60m = MeteorologicalCalculator.wspd_wdir_to_uv(wspd_60m, wdir_60m)

        return u_60m, v_60m


# ================= MICAPS4格式保存类 =================
class MICAPS4Writer:
    """MICAPS第4类数据格式写入器 - 二进制格式"""

    # 要素到MICAPS要素名的映射
    ELEMENT_MAP = {
        "WIND": "10WIND",
        "WIND100": "100WIND",
        "WIND60": "60WIND",
        "GUST": "10GUST",
        "TEM": "2MT",
        "PRS": "SFC_PR",
        "DPT": "2MDPT",
        "RH": "2MRH",
        "MN2T6": "MN2T6",
        "MX2T6": "MX2T6",
        "VIS": "VIS",
        "PRE": "PRE",
        "TPE": "TPE",
        "TCC": "TCC"
    }

    # 矢量要素的level值映射（高度对应的数值）
    VECTOR_LEVEL_MAP = {
        "WIND": 10.0,
        "WIND100": 100.0,
        "WIND60": 60.0
    }

    # 标量要素的level值映射（气象层次）
    SCALAR_LEVEL_MAP = {
        "GUST": 10.0,      # 10m阵风
        "TEM": 1000.0,      # 2m温度，通常用1000表示地面
        "PRS": 1000.0,      # 地面气压
        "DPT": 1000.0,      # 2m露点温度
        "RH": 1000.0,       # 2m相对湿度
        "MN2T6": 1000.0,   # 6h最低温度
        "MX2T6": 1000.0,    # 6h最高温度
        "VIS": 1000.0,      # 地面能见度
        "PRE": 1000.0,      # 地面降水
        "TPE": 1000.0,      # 总降水量
        "TCC": 1000.0       # 总云量
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

            # 根据要素类型设置正确的level值
            if element not in MICAPS4Writer.SCALAR_LEVEL_MAP:
                # 如果不在映射表中，使用默认值1000.0（地面）
                level = 1000.0
            else:
                level = MICAPS4Writer.SCALAR_LEVEL_MAP[element]

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

                # ============= 12. period (时效) =============
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

                # ============= 25. 数据 =============
                # 处理NaN值
                data_clean = data.copy()
                nan_mask = np.isnan(data_clean)
                if np.any(nan_mask):
                    data_clean[nan_mask] = 9999.0

                # 数据按行优先写入（纬度优先）
                # 展平数组并确保是32位浮点数
                data_flat = data_clean.astype(np.float32).ravel()

                # 转换为二进制
                for i in range(0, len(data_flat), 10000):
                    start_idx = i
                    end_idx = min(i + 10000, len(data_flat))
                    f.write(data_flat[start_idx:end_idx].tobytes())

            return True

        except Exception as e:
            return False

    @staticmethod
    def write_micaps4_vector_file(wspd_data: np.ndarray, wdir_data: np.ndarray,
                                  lats: np.ndarray, lons: np.ndarray,
                                  base_time: datetime, forecast_hour: int,
                                  output_path: str, element: str = "WIND",
                                  model_name: str = "ECMWF", height: str = "10m",
                                  description: str = "", timezone_shift: timedelta = None) -> bool:
        """
        写入MICAPS4矢量格式文件（type=11）

        Parameters
        ----------
        wspd_data : np.ndarray
            风速数据
        wdir_data : np.ndarray
            风向数据
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
        height : str
            高度描述（如"10m", "100m", "60m"）
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
            if len(wspd_data.shape) != 2 or len(wdir_data.shape) != 2:
                raise ValueError(f"数据必须是2D数组，wspd形状: {wspd_data.shape}, wdir形状: {wdir_data.shape}")

            if wspd_data.shape != wdir_data.shape:
                raise ValueError(f"风速和风向数据形状不匹配: {wspd_data.shape} vs {wdir_data.shape}")

            if wspd_data.shape != (len(lats), len(lons)):
                raise ValueError(f"数据形状{wspd_data.shape}与网格形状({len(lats)}, {len(lons)})不匹配")

            n_lat, n_lon = len(lats), len(lons)

            # 确保纬度从南到北（升序），经度从西到东（升序）
            if lats[0] > lats[-1]:  # 降序
                lats = lats[::-1]
                wspd_data = wspd_data[::-1, :]
                wdir_data = wdir_data[::-1, :]

            if lons[0] > lons[-1]:  # 降序
                lons = lons[::-1]
                wspd_data = wspd_data[:, ::-1]
                wdir_data = wdir_data[:, ::-1]

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

                # ============= 2. type: 11为矢量数据 =============
                data_type = 11  # 矢量数据
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
                desc = description or f"{height} wind speed(m/s) and direction(deg)"
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
                # 根据要素类型获取正确的level值
                level = MICAPS4Writer.VECTOR_LEVEL_MAP.get(element, 0.0)
                f.write(np.float32(level).tobytes())

                # ============= 7-10. 起报日期和时间 =============
                f.write(np.int32(local_time.year).tobytes())
                f.write(np.int32(local_time.month).tobytes())
                f.write(np.int32(local_time.day).tobytes())
                f.write(np.int32(local_time.hour).tobytes())

                # ============= 11. 时区 =============
                f.write(np.int32(8).tobytes())  # 北京时区

                # ============= 12. period (时效) =============
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
                wspd_clean = wspd_data.copy()
                wdir_clean = wdir_data.copy()

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
            print(f"✗ 写入MICAPS4矢量文件失败: {str(e)}")
            import traceback
            traceback.print_exc()
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
        bool, Dict[str, np.ndarray], List[datetime], str, np.ndarray]:
        """处理标量数据（从NetCDF文件读取）

        Returns
        -------
        bool
            是否成功
        Dict[str, np.ndarray]
            处理后的数据字典
        List[datetime]
            时间序列（北京时）
        str
            最终输出文件路径
        np.ndarray
            预报时效数组（0, 1, 2, ...）
        """
        try:
            self.logger.info(f"处理标量要素: {element}")

            # 获取要素配置
            element_config = Config.ELEMENTS[element]

            # 1. 读取NetCDF数据
            start_time = time.time()

            with Dataset(nc_file, 'r') as ds:
                # 根据nc_mapping获取变量名
                if "nc_mapping" in element_config:
                    nc_mapping = element_config["nc_mapping"]
                    input_var = nc_mapping.get("input_var")

                    # 如果没有指定input_var，尝试自动匹配
                    if input_var is None:
                        # 获取数据变量（排除坐标变量）
                        data_vars = [v for v in ds.variables.keys()
                                      if v not in ['time', 'latitude', 'longitude', 'lat', 'lon', 'level']]
                        if not data_vars:
                            self.logger.error("NetCDF文件中未找到数据变量")
                            return False, None, None
                        var_name = data_vars[0]
                    else:
                        var_name = input_var
                else:
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
                needs_flip_lat = lat_src[-1] < lat_src[0]
                if needs_flip_lat:
                    self.logger.info("源纬度是降序，将反转纬度坐标")
                    # 先不反转，等数据读取后再统一处理

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

                        # 如果纬度需要反转，同时反转数据的纬度维度
                        if needs_flip_lat:
                            self.logger.info("反转纬度坐标和数据")
                            lat_src = lat_src[::-1]
                            all_data = all_data[:, ::-1, :]

                        # 创建时间-数据对
                        time_data_pairs = list(zip(steps, range(len(steps))))

                        # 按时间排序
                        time_data_pairs.sort(key=lambda x: x[0])

                        # 重新提取排序后的steps和数据
                        steps_sorted = [item[0] for item in time_data_pairs]
                        indices_sorted = [item[1] for item in time_data_pairs]

                        # 按排序后的索引重新读取数据
                        data_cube = all_data[indices_sorted, :, :]

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

            # 3. 判断是否为降水要素
            is_precipitation = element in ["TPE", "PRE"]

            # 4. 时间插值（降水要素和非降水要素处理方式不同）
            start_time = time.time()

            if is_precipitation:
                # 降水要素：累积降水 → 时段降水 → 小时降水
                self.logger.info("检测到降水要素，使用降水专用处理逻辑")
                data_1h, hours_new = self._process_precipitation_time_interpolation(
                    data_interp, steps
                )
                # 生成北京时时间序列
                times_bjt = [base_time_utc + timedelta(hours=float(h)) + Config.TIMEZONE_SHIFT
                             for h in hours_new]
            else:
                # 非降水要素：普通时间插值
                # 先进行单位转换
                if element_config.get("conversion") == "K_to_C":
                    self.logger.info(f"单位转换: 开尔文(K) → 摄氏度(°C)")
                    data_interp = MeteorologicalCalculator.kelvin_to_celsius(data_interp)

                # 检查时间间隔是否一致
                intervals = [steps[i+1] - steps[i] for i in range(len(steps)-1)]
                interval_set = set(intervals)
                is_uniform_interval = len(interval_set) == 1

                # 分段处理
                data_1h_segments = []
                hour_segments = []

                if is_uniform_interval:
                    # 所有时间点间隔一致，直接处理不分段
                    interval = intervals[0]
                    self.logger.info(f"时间间隔一致（{interval}h），直接插值不分段")

                    min_hour_data = min(steps)
                    max_hour = max(steps)

                    # 从0开始生成时间序列
                    hours_new = np.arange(0, max_hour + 1, 1.0)

                    # 在有数据的时间点进行插值
                    f_data = interp1d(steps, data_interp, axis=0, kind='linear',
                                     bounds_error=False, fill_value=np.nan)
                    data_1h = f_data(hours_new)

                    if min_hour_data > 0:
                        self.logger.info(f"  {min_hour_data}-{max_hour}h: {len(steps)}个原始时次（{interval}h间隔）→ {len(hours_new)}个1小时间隔时次（前{min_hour_data}h为NaN）")
                    else:
                        self.logger.info(f"  {min_hour_data}-{max_hour}h: {len(steps)}个原始时次（{interval}h间隔）→ {len(hours_new)}个1小时间隔时次")

                    data_1h_segments = [data_1h]
                    hour_segments = [hours_new]
                else:
                    # 时间间隔不一致，分段处理
                    self.logger.info(f"时间间隔不一致（{set(intervals)}），分段处理")

                    # 根据时效分两段
                    steps_3h = [step for step in steps if step <= 72]
                    steps_6h = [step for step in steps if step > 72 and step <= 240]

                    # 第一段: 0-72小时 (3小时间隔)
                    if steps_3h:
                        indices_3h = [i for i, step in enumerate(steps) if step in steps_3h]
                        data_3h = data_interp[indices_3h]

                        hours_3h = steps_3h
                        # 确保从0小时开始到72小时
                        min_hour_data = min(hours_3h)
                        max_hour = min(max(hours_3h), 72)

                        # 从0开始生成时间序列
                        hours_new_3h = np.arange(0, max_hour + 1, 1.0)

                        # 只在有数据的时间点进行插值
                        f_data_3h = interp1d(hours_3h, data_3h, axis=0, kind='linear',
                                            bounds_error=False, fill_value=np.nan)
                        data_1h_3h = f_data_3h(hours_new_3h)

                        # 对于没有原始数据的时间点（如0-2小时），保持为NaN
                        if min_hour_data > 0:
                            self.logger.info(f"  0-72小时: {len(hours_3h)}个原始时次（从{min_hour_data}h开始）→ {len(hours_new_3h)}个1小时间隔时次")
                        else:
                            self.logger.info(f"  0-72小时: {len(hours_3h)}个原始时次 → {len(hours_new_3h)}个1小时间隔时次")

                        data_1h_segments.append(data_1h_3h)
                        hour_segments.append(hours_new_3h)

                    # 第二段: 72-240小时 (6小时间隔)
                    if steps_6h:
                        indices_6h = [i for i, step in enumerate(steps) if step in steps_6h]
                        data_6h = data_interp[indices_6h]

                        hours_6h = steps_6h
                        max_forecast_hour = max(steps_6h)
                        start_hour_6h = 73
                        end_hour_6h = min(max_forecast_hour, 240)

                        if start_hour_6h <= end_hour_6h:
                            # 确保从73小时开始到最大时效
                            min_hour_data = min(hours_6h)
                            max_hour = min(max(hours_6h), end_hour_6h)

                            # 从73开始生成时间序列（如果原始数据从更晚开始，前面会自动填充NaN）
                            hours_new_6h = np.arange(start_hour_6h, max_hour + 1, 1.0)

                            # 只在有数据的时间点进行插值
                            f_data_6h = interp1d(hours_6h, data_6h, axis=0, kind='linear',
                                                bounds_error=False, fill_value=np.nan)
                            data_1h_6h = f_data_6h(hours_new_6h)

                            if min_hour_data > start_hour_6h:
                                self.logger.info(
                                    f"  73-{end_hour_6h}小时: {len(hours_6h)}个原始时次（从{min_hour_data}h开始）→ {len(hours_new_6h)}个1小时间隔时次")
                            else:
                                self.logger.info(
                                    f"  73-{end_hour_6h}小时: {len(hours_6h)}个原始时次 → {len(hours_new_6h)}个1小时间隔时次")

                            data_1h_segments.append(data_1h_6h)
                            hour_segments.append(hours_new_6h)
                        else:
                            self.logger.warning(f"  72-240小时段无有效数据")

                # 合并所有分段
                if data_1h_segments:
                    data_1h = np.concatenate(data_1h_segments, axis=0)
                    hours_new = np.concatenate(hour_segments, axis=0)

                    # 添加调试信息
                    self.logger.info(f"合并后的数据形状: {data_1h.shape}")
                    self.logger.info(f"数据范围: {np.nanmin(data_1h):.2f} - {np.nanmax(data_1h):.2f}")
                    nan_count = np.count_nonzero(np.isnan(data_1h))
                    self.logger.info(f"NaN数量: {nan_count}/{data_1h.size}")

                    # 生成北京时时间序列
                    times_bjt = [base_time_utc + timedelta(hours=float(h)) + Config.TIMEZONE_SHIFT
                                 for h in hours_new]

                    time_interp_time = time.time() - start_time
                    self.logger.info(f"时间插值完成: {time_interp_time:.1f}秒, 总时次: {len(times_bjt)}")
                else:
                    self.logger.error("时间插值失败：没有生成有效数据")
                    return False, None, None

            # 获取变量名（在外部作用域中定义）
            var_name = element_config["output_vars"][0]

            # 5. 写入NetCDF
            start_time = time.time()

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with Dataset(output_path, 'w') as nc:
                # 创建维度
                nc.createDimension('time', len(times_bjt))
                nc.createDimension('lat', self.n_lat_dst)
                nc.createDimension('lon', self.n_lon_dst)

                # 坐标变量
                time_var = nc.createVariable('time', 'i4', ('time',))
                lat_var = nc.createVariable('lat', 'f4', ('lat',))
                lon_var = nc.createVariable('lon', 'f4', ('lon',))

                # 数据变量
                data_var = nc.createVariable(var_name, 'f4', ('time', 'lat', 'lon'),
                                             zlib=True, complevel=1)

                # 设置数据
                lat_var[:] = self.lat_dst
                lat_var.units = 'degrees_north'
                lat_var.long_name = 'latitude (south to north)'

                lon_var[:] = self.lon_dst
                lon_var.units = 'degrees_east'

                time_var.units = 'hours since 1970-01-01 00:00:00'
                time_var.calendar = 'gregorian'
                time_var.time_zone = 'UTC+8'
                time_var[:] = date2num(times_bjt, time_var.units, time_var.calendar)

                data_var[:] = data_1h
                data_var.units = element_config["units"][var_name]
                data_var.long_name = element_config["description"]

                # 添加单位转换说明（如果需要）
                if element_config.get("conversion") == "K_to_C":
                    data_var.comment = 'Original data in Kelvin, converted to Celsius'
                elif is_precipitation:
                    data_var.comment = 'Hourly precipitation calculated from accumulated precipitation (米 → 毫米)'

                # 添加全局属性
                if is_precipitation:
                    nc.time_interpolation = '累积降水转换为时段降水，均匀分配到每小时'
                else:
                    nc.time_interpolation = '分段线性插值: 0-72小时(3h→1h), 73-240小时(6h→1h)'

                nc.title = f'ECMWF {element_config["description"]} (Beijing Time)'
                nc.source = 'ECMWF Forecast'
                nc.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                nc.forecast_start_time_utc = base_time_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
                nc.forecast_start_time_bjt = Config.utc_to_bjt_str(base_time_utc)
                nc.latitude_order = 'south to north (ascending)'

            write_time = time.time() - start_time
            file_size_mb = os.path.getsize(output_path) / 1024 / 1024
            self.logger.info(f"NetCDF文件写入完成: {write_time:.1f}秒, 大小: {file_size_mb:.1f} MB")

            # 检查是否是临时文件（.tmp后缀），如果是则重命名为最终文件名
            import shutil
            final_output_path = output_path
            if output_path.endswith('.tmp'):
                final_output_path = output_path[:-4]  # 去掉 .tmp 后缀
                self.logger.info(f"临时文件处理完成，重命名: {output_path} -> {final_output_path}")
                shutil.move(output_path, final_output_path)
            else:
                self.logger.info(f"文件已直接写入: {output_path}")

            # 返回数据和最终路径
            result_data = {var_name: data_1h}

            # 内存清理（注意：times_bjt 在 return 语句中使用，不能删除）
            del data_1h
            gc.collect()

            return True, result_data, times_bjt, final_output_path, hours_new

        except Exception as e:
            self.logger.error(f"处理标量数据失败: {str(e)}")
            return False, None, None, None, None

    def _process_precipitation_time_interpolation(
        self, accum_data: np.ndarray, steps: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        降水时间插值：累积降水 → 时段降水 → 小时降水

        Parameters
        ----------
        accum_data : np.ndarray
            空间插值后的累积降水数据，单位：米
        steps : List[int]
            预报时效列表（小时）

        Returns
        -------
        data_1h : np.ndarray
            1小时间隔的降水数据，单位：毫米
        hours_new : np.ndarray
            时间序列（0到最大时效，每小时1个）
        """
        # 1. 单位转换：米 → 毫米
        accum_data_mm = accum_data * 1000.0
        self.logger.info(f"单位转换: 米 → 毫米")

        # 2. 计算时段降水（后一时刻 - 前一时刻）
        n_times = len(steps)
        n_intervals = n_times - 1

        interval_precip = np.empty((n_intervals, accum_data_mm.shape[1], accum_data_mm.shape[2]), dtype=np.float32)

        for i in range(n_intervals):
            # 时段降水 = 后一时刻累积 - 前一时刻累积
            interval_precip[i] = accum_data_mm[i+1] - accum_data_mm[i]

            # 检查负值（由于数值误差可能出现微小负值）
            negative_mask = interval_precip[i] < -0.001  # 允许微小负值
            negative_count = np.sum(negative_mask)
            if negative_count > 0:
                self.logger.info(f"  时段{i}（{steps[i]}-{steps[i+1]}h）: 发现{negative_count}个负值，设为0")
                interval_precip[i][negative_mask] = 0.0

        # 3. 均匀分配到小时
        max_hour = max(steps)
        n_lat, n_lon = interval_precip.shape[1], interval_precip.shape[2]
        hourly_precip = np.zeros((max_hour, n_lat, n_lon), dtype=np.float32)

        # 分配逻辑
        for i in range(n_intervals):
            interval = steps[i+1] - steps[i]
            start_hour = steps[i]

            # 均匀分配
            hourly_amount = interval_precip[i] / float(interval)

            # 分配到小时
            for hour_offset in range(interval):
                hour_idx = start_hour + hour_offset
                if hour_idx < max_hour:
                    hourly_precip[hour_idx] = hourly_amount

        # 4. 在数组最前面插入一个全是 nan 的时间点（起报时间的累计降水初始值）
        hourly_precip = np.insert(hourly_precip, 0, np.nan, axis=0)

        # 5. 生成时间序列（从0到max_hour，共max_hour+1个点，与其他要素如气温保持一致）
        hours_new = np.arange(0, max_hour + 1, 1.0)

        # 输出处理信息
        self.logger.info(f"降水时间处理完成:")
        self.logger.info(f"  原始时次: {len(steps)}个（{steps[:5]}...{steps[-5:]}）")
        self.logger.info(f"  时段数: {n_intervals}个")
        self.logger.info(f"  小时降水: {len(hourly_precip)}个（0-{max_hour}h，第0点为起报时间初始值NaN）")
        self.logger.info(f"  数据范围: {np.nanmin(hourly_precip):.4f} - {np.nanmax(hourly_precip):.4f} mm")

        # 内存清理
        del accum_data_mm, interval_precip
        gc.collect()

        return hourly_precip, hours_new

    def _find_uv_files(self, nc_file: str, element: str) -> Dict[str, str]:
        """
        根据输入文件和要素类型，查找对应的U/V文件对

        Parameters
        ----------
        nc_file : str
            输入的NetCDF文件路径
        element : str
            要素名称 (WIND, WIND100, WIND60)

        Returns
        -------
        dict
            包含 u 和 v 文件路径的字典
        """
        nc_files = {"u": None, "v": None}
        file_dir = os.path.dirname(nc_file)
        basename = os.path.basename(nc_file)

        # 根据文件名推断U/V文件对
        if element == "WIND":
            # 10m风场，需要10U和10V
            if "10U" in basename or "10u" in basename:
                u_file = nc_file
                # 将10U替换为10V
                v_basename = basename.replace("10U", "10V").replace("10u", "10v")
                v_file = os.path.join(file_dir, v_basename)
            elif "10V" in basename or "10v" in basename:
                v_file = nc_file
                # 将10V替换为10U
                u_basename = basename.replace("10V", "10U").replace("10v", "10u")
                u_file = os.path.join(file_dir, u_basename)
            else:
                # 尝试在目录中查找
                time_pattern = self._extract_time_pattern(basename)
                u_basename = f"ECMFC1D_10U_1_{time_pattern}_GLB_1.nc"
                v_basename = f"ECMFC1D_10V_1_{time_pattern}_GLB_1.nc"
                u_file = os.path.join(file_dir, u_basename)
                v_file = os.path.join(file_dir, v_basename)

            nc_files["u"] = u_file if os.path.exists(u_file) else None
            nc_files["v"] = v_file if os.path.exists(v_file) else None

        elif element in ["WIND100", "WIND60"]:
            # 100m风场（WIND60也基于100m数据），需要100U和100V
            if "100U" in basename or "100u" in basename:
                u_file = nc_file
                # 将100U替换为100V
                v_basename = basename.replace("100U", "100V").replace("100u", "100v")
                v_file = os.path.join(file_dir, v_basename)
            elif "100V" in basename or "100v" in basename:
                v_file = nc_file
                # 将100V替换为100U
                u_basename = basename.replace("100V", "100U").replace("100v", "100u")
                u_file = os.path.join(file_dir, u_basename)
            else:
                # 尝试在目录中查找
                time_pattern = self._extract_time_pattern(basename)
                u_basename = f"ECMFC1D_100U_1_{time_pattern}_GLB_1.nc"
                v_basename = f"ECMFC1D_100V_1_{time_pattern}_GLB_1.nc"
                u_file = os.path.join(file_dir, u_basename)
                v_file = os.path.join(file_dir, v_basename)

            nc_files["u"] = u_file if os.path.exists(u_file) else None
            nc_files["v"] = v_file if os.path.exists(v_file) else None

        return nc_files

    def _extract_time_pattern(self, filename: str) -> str:
        """
        从文件名中提取时间模式

        Parameters
        ----------
        filename : str
            文件名

        Returns
        -------
        str
            时间模式字符串（如 2026010100）
        """
        # 使用正则表达式匹配连续的10位数字
        import re
        match = re.search(r'(\d{10})', filename)
        if match:
            return match.group(1)
        return ""

    def _process_wind_data(self, element: str, nc_files: Dict[str, str],
                         base_time_utc: datetime, output_path: str) -> Tuple[
        bool, Dict[str, np.ndarray], List[datetime], np.ndarray]:
        """
        处理风场数据（需要成对的U/V分量NetCDF文件）

        Parameters
        ----------
        element : str
            要素名称 (WIND, WIND100, WIND60)
        nc_files : dict
            包含 u 和 v 文件的字典
        base_time_utc : datetime
            基准时间（UTC）
        output_path : str
            输出文件路径

        Returns
        -------
        tuple
            (是否成功, 结果数据字典, 时间列表, 预报时效数组)
        """
        try:
            self.logger.info(f"处理风场要素: {element}")

            # 获取要素配置
            element_config = Config.ELEMENTS[element]

            # 检查输入文件
            u_file = nc_files.get("u")
            v_file = nc_files.get("v")

            if not u_file or not v_file:
                self.logger.error(f"风场处理需要U和V分量文件，缺少: u={u_file}, v={v_file}")
                return False, None, None

            # 检查文件是否存在
            if not os.path.exists(u_file):
                self.logger.error(f"U分量文件不存在: {u_file}")
                return False, None, None
            if not os.path.exists(v_file):
                self.logger.error(f"V分量文件不存在: {v_file}")
                return False, None, None

            # 1. 读取U和V数据
            start_time = time.time()
            self.logger.info(f"读取U分量: {os.path.basename(u_file)}")
            self.logger.info(f"读取V分量: {os.path.basename(v_file)}")

            with Dataset(u_file, 'r') as ds_u, Dataset(v_file, 'r') as ds_v:
                # 读取U数据
                u_data_cube, steps_u, lat_src_u, lon_src_u = self._read_netcdf_wind_data(ds_u, "u")
                # 读取V数据
                v_data_cube, steps_v, lat_src_v, lon_src_v = self._read_netcdf_wind_data(ds_v, "v")

                # 检查时间是否匹配
                if steps_u != steps_v:
                    self.logger.error(f"U和V的时间不匹配: U={steps_u[:5]}, V={steps_v[:5]}")
                    return False, None, None

                # 使用U的经纬度（假设U和V网格相同）
                lat_src = lat_src_u
                lon_src = lon_src_u

            read_time = time.time() - start_time
            self.logger.info(f"数据读取完成: {read_time:.1f}秒, 时次: {len(steps_u)}")
            self.logger.info(f"预报时次范围: {min(steps_u) if steps_u else 'None'} - {max(steps_u) if steps_u else 'None'} 小时")
            self.logger.info(f"前5个时次: {steps_u[:5] if steps_u else 'None'}")

            # 2. 空间插值
            start_time = time.time()

            lon2d, lat2d = np.meshgrid(self.lon_dst, self.lat_dst)
            points = np.column_stack([lat2d.ravel(), lon2d.ravel()])

            u_interp = np.empty((len(steps_u), self.n_lat_dst, self.n_lon_dst), dtype=np.float32)
            v_interp = np.empty_like(u_interp)

            for i in range(len(steps_u)):
                interp_func = RegularGridInterpolator(
                    (lat_src, lon_src),
                    u_data_cube[i],
                    bounds_error=False,
                    fill_value=np.nan
                )
                u_interp[i] = interp_func(points).reshape(self.n_lat_dst, self.n_lon_dst)

                interp_func = RegularGridInterpolator(
                    (lat_src, lon_src),
                    v_data_cube[i],
                    bounds_error=False,
                    fill_value=np.nan
                )
                v_interp[i] = interp_func(points).reshape(self.n_lat_dst, self.n_lon_dst)

            interp_time = time.time() - start_time
            self.logger.info(f"空间插值完成: {interp_time:.1f}秒")

            # 3. 对于WIND60，从100m计算60m风场
            if element == "WIND60":
                start_time = time.time()
                self.logger.info("从100米风场计算60米风场...")

                u_60m_interp = np.empty_like(u_interp)
                v_60m_interp = np.empty_like(v_interp)

                for i in range(len(steps_u)):
                    u_60m_interp[i], v_60m_interp[i] = MeteorologicalCalculator.calculate_60m_wind_from_100m(
                        u_interp[i], v_interp[i], method='power_law'
                    )

                u_interp = u_60m_interp
                v_interp = v_60m_interp

                calc_time = time.time() - start_time
                self.logger.info(f"60米风场计算完成: {calc_time:.1f}秒")

            # 4. 时间插值（分段处理）
            start_time = time.time()

            # 根据时效分两段：0-72h(3h), 72-240h(6h)
            steps_3h = [step for step in steps_u if step <= 72]
            steps_6h = [step for step in steps_u if step > 72 and step <= 240]

            # 分段处理
            u_1h_segments = []
            v_1h_segments = []
            hour_segments = []

            # 第一段: 0-72小时 (3小时间隔)
            if steps_3h:
                indices_3h = [i for i, step in enumerate(steps_u) if step in steps_3h]
                u_3h = u_interp[indices_3h]
                v_3h = v_interp[indices_3h]

                hours_3h = steps_3h
                max_hour = min(max(hours_3h), 72)

                # 从0开始生成时间序列
                hours_new_3h = np.arange(0, max_hour + 1, 1.0)

                f_u_3h = interp1d(hours_3h, u_3h, axis=0, kind='linear',
                                    bounds_error=False, fill_value=np.nan)
                f_v_3h = interp1d(hours_3h, v_3h, axis=0, kind='linear',
                                    bounds_error=False, fill_value=np.nan)

                u_1h_3h = f_u_3h(hours_new_3h)
                v_1h_3h = f_v_3h(hours_new_3h)

                u_1h_segments.append(u_1h_3h)
                v_1h_segments.append(v_1h_3h)
                hour_segments.append(hours_new_3h)

                self.logger.info(f"  0-72小时: {len(hours_3h)}个原始时次 → {len(hours_new_3h)}个1小时间隔时次")

            # 第二段: 72-240小时 (6小时间隔)
            if steps_6h:
                indices_6h = [i for i, step in enumerate(steps_u) if step in steps_6h]
                u_6h = u_interp[indices_6h]
                v_6h = v_interp[indices_6h]

                hours_6h = steps_6h
                max_forecast_hour = max(steps_6h)
                start_hour_6h = 73
                end_hour_6h = min(max_forecast_hour, 240)

                if start_hour_6h <= end_hour_6h:
                    max_hour = min(max(hours_6h), end_hour_6h)

                    # 从73开始生成时间序列
                    hours_new_6h = np.arange(start_hour_6h, max_hour + 1, 1.0)

                    f_u_6h = interp1d(hours_6h, u_6h, axis=0, kind='linear',
                                        bounds_error=False, fill_value=np.nan)
                    f_v_6h = interp1d(hours_6h, v_6h, axis=0, kind='linear',
                                        bounds_error=False, fill_value=np.nan)

                    u_1h_6h = f_u_6h(hours_new_6h)
                    v_1h_6h = f_v_6h(hours_new_6h)

                    u_1h_segments.append(u_1h_6h)
                    v_1h_segments.append(v_1h_6h)
                    hour_segments.append(hours_new_6h)

                    self.logger.info(
                        f"  73-{end_hour_6h}小时: {len(hours_6h)}个原始时次 → {len(hours_new_6h)}个1小时间隔时次")

            # 合并所有分段
            if u_1h_segments:
                u_1h = np.concatenate(u_1h_segments, axis=0)
                v_1h = np.concatenate(v_1h_segments, axis=0)
                hours_new = np.concatenate(hour_segments, axis=0)

                time_interp_time = time.time() - start_time
                self.logger.info(f"时间插值完成: {time_interp_time:.1f}秒, 总时次: {len(hours_new)}")
            else:
                self.logger.error("时间插值失败：没有生成有效数据")
                return False, None, None

            # 5. 生成北京时时间序列
            times_bjt = [base_time_utc + timedelta(hours=float(h)) + Config.TIMEZONE_SHIFT
                         for h in hours_new]

            # 6. 写入NetCDF
            start_time = time.time()

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with Dataset(output_path, 'w') as nc:
                # 创建维度
                nc.createDimension('time', len(times_bjt))
                nc.createDimension('lat', self.n_lat_dst)
                nc.createDimension('lon', self.n_lon_dst)

                # 坐标变量
                time_var = nc.createVariable('time', 'i4', ('time',))
                lat_var = nc.createVariable('lat', 'f4', ('lat',))
                lon_var = nc.createVariable('lon', 'f4', ('lon',))

                # 数据变量：只输出U/V分量
                u_var = nc.createVariable(f'u_{self._get_height_suffix(element)}', 'f4',
                                        ('time', 'lat', 'lon'),
                                        zlib=True, complevel=1)
                v_var = nc.createVariable(f'v_{self._get_height_suffix(element)}', 'f4',
                                        ('time', 'lat', 'lon'),
                                        zlib=True, complevel=1)

                # 设置数据
                lat_var[:] = self.lat_dst
                lat_var.units = 'degrees_north'
                lat_var.long_name = 'latitude (south to north)'

                lon_var[:] = self.lon_dst
                lon_var.units = 'degrees_east'

                time_var.units = 'hours since 1970-01-01 00:00:00'
                time_var.calendar = 'gregorian'
                time_var.time_zone = 'UTC+8'
                time_var[:] = date2num(times_bjt, time_var.units, time_var.calendar)

                height_desc = self._get_height_description(element)

                u_var[:] = u_1h
                u_var.units = 'm s-1'
                u_var.long_name = f'{height_desc} wind u-component'

                v_var[:] = v_1h
                v_var.units = 'm s-1'
                v_var.long_name = f'{height_desc} wind v-component'

                # 添加全局属性
                nc.title = f'ECMWF {height_desc} Wind U/V Components (Beijing Time)'
                nc.source = 'ECMWF Forecast'
                nc.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                nc.forecast_start_time_utc = base_time_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
                nc.forecast_start_time_bjt = Config.utc_to_bjt_str(base_time_utc)
                nc.latitude_order = 'south to north (ascending)'
                nc.time_interpolation = '分段线性插值: 0-72小时(3h→1h), 73-240小时(6h→1h)'

                if element == "WIND60":
                    nc.wind_height_calculation = '从100米风场计算，使用幂律方法'
                    nc.power_law_exponent = str(Config.WIND_PROFILE_PARAMS["power_law_exponent"])

            write_time = time.time() - start_time
            self.logger.info(f"NetCDF文件写入完成: {write_time:.1f}秒")

            # 7. 构建返回数据
            result_data = {
                f"u_{self._get_height_suffix(element)}": u_1h,
                f"v_{self._get_height_suffix(element)}": v_1h
            }

            # 内存清理
            del u_1h, v_1h
            gc.collect()

            return True, result_data, times_bjt, hours_new

        except Exception as e:
            self.logger.error(f"处理风场数据失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, None, None

    def _read_netcdf_wind_data(self, ds, var_type):
        """
        从NetCDF数据集读取风场数据

        Parameters
        ----------
        ds : netCDF4.Dataset
            NetCDF数据集
        var_type : str
            变量类型 ("u" 或 "v")

        Returns
        -------
        tuple
            (数据立方体, 时次列表, 纬度, 经度)
        """
        # 查找数据变量
        data_vars = [v for v in ds.variables.keys()
                     if v not in ['time', 'latitude', 'longitude', 'lat', 'lon', 'level']]
        if not data_vars:
            raise ValueError("NetCDF文件中未找到数据变量")

        var_name = data_vars[0]

        # 读取经纬度
        if 'latitude' in ds.variables:
            lat_var = ds.variables['latitude']
            lon_var = ds.variables['longitude']
        elif 'lat' in ds.variables:
            lat_var = ds.variables['lat']
            lon_var = ds.variables['lon']
        else:
            raise ValueError("NetCDF文件中未找到经纬度坐标变量")

        lat_src = lat_var[:]
        lon_src = lon_var[:]

        # 确保纬度升序
        needs_flip_lat = lat_src[-1] < lat_src[0]

        # 读取时间
        time_var = ds.variables['time']
        time_values = time_var[:].flatten()

        # 解析时间单位获取基准时间
        time_base = datetime(1970, 1, 1)  # 默认基准时间
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
        except Exception:
            pass

        # 转换为预报时次（相对于00时）
        steps = []
        data_all = ds.variables[var_name][:]  # shape: (time, lat, lon)

        # 如果纬度需要反转
        if needs_flip_lat:
            lat_src = lat_src[::-1]
            data_all = data_all[:, ::-1, :]

        for t in time_values:
            try:
                forecast_time = time_base + timedelta(hours=float(t))
                # time_values本身就是相对于基准时间的预报时效，直接使用
                forecast_hour = int(t)
                steps.append(forecast_hour)
            except:
                pass

        steps = sorted(set(steps))
        # 按时间排序
        time_data_pairs = list(zip(steps, range(len(steps))))
        time_data_pairs.sort(key=lambda x: x[0])

        steps_sorted = [item[0] for item in time_data_pairs]
        indices_sorted = [item[1] for item in time_data_pairs]

        data_cube = data_all[indices_sorted, :, :]

        return data_cube, steps_sorted, lat_src, lon_src

    def _get_height_suffix(self, element: str) -> str:
        """根据要素获取高度后缀"""
        if element == "WIND":
            return "10m"
        elif element == "WIND100":
            return "100m"
        elif element == "WIND60":
            return "60m"
        else:
            return ""

    def _get_height_description(self, element: str) -> str:
        """根据要素获取高度描述"""
        if element == "WIND":
            return "10 meter"
        elif element == "WIND100":
            return "100 meter"
        elif element == "WIND60":
            return "60 meter"
        else:
            return ""

    def process_element(self, element: str, nc_file: str,
                        output_dir: str = None, base_time: datetime = None,
                        output_filename: str = None, skip_existing: bool = True,
                        save_micaps4: bool = None, micaps4_output_dir: str = None) -> Tuple[
        bool, str, Dict]:
        """
        处理单个要素数据

        Parameters
        ----------
        element : str
            要素名称（WIND, GUST, TEM, PRS, DPT, RH）
        nc_file : str
            输入的NetCDF文件路径
        output_dir : str, optional
            输出目录
        base_time : datetime, optional
            基准时间（UTC）
        output_filename : str, optional
            输出文件名
        skip_existing : bool
            是否跳过已存在的输出文件
        save_micaps4 : bool
            是否保存MICAPS4格式
        micaps4_output_dir : str
            MICAPS4输出目录

        Returns
        -------
        tuple
            (是否成功, NetCDF输出文件路径, MICAPS4文件信息字典)
        """
        total_start = time.time()
        success = False
        output_path = None
        micaps4_files = {}

        # 确定是否保存MICAPS4
        if save_micaps4 is None:
            save_micaps4 = self.save_micaps4

        # 确定MICAPS4输出目录
        if micaps4_output_dir is None:
            micaps4_output_dir = self.micaps4_output_dir or output_dir

        try:
            # 1. 检查要素配置
            if element not in Config.ELEMENTS:
                self.logger.error(f"不支持的元素: {element}")
                self.logger.error(f"支持的元素: {list(Config.ELEMENTS.keys())}")
                return False, None, {}

            element_config = Config.ELEMENTS[element]
            self.logger.info(f"开始处理要素: {element} - {element_config['description']}")

            # 2. 确定基准时间
            if base_time is None:
                # 从文件名解析时间
                base_time = Config.parse_time_from_filename(nc_file)

            if base_time is None:
                self.logger.error("无法从文件名解析时间，请提供base_time参数")
                return False, None, {}

            # 3. 确定输出路径
            if output_dir is None:
                output_dir = os.path.dirname(nc_file)

            if output_filename is None:
                # 使用nc_mapping中的output_filename
                if "nc_mapping" in element_config and "output_filename" in element_config["nc_mapping"]:
                    output_filename_base = element_config["nc_mapping"]["output_filename"]
                    bjt_time_str = Config.utc_to_bjt_str(base_time)
                    output_filename = f"{output_filename_base}_0p01_1h_BJT_{bjt_time_str}.nc"
                else:
                    # 默认格式
                    bjt_time_str = Config.utc_to_bjt_str(base_time)
                    output_filename = Config.OUTPUT_FILENAME_FORMAT.format(
                        element=element.lower(), time_str=bjt_time_str
                    )

            # 按照要素和起报时间分类存放
            bjt_time_str = Config.utc_to_bjt_str(base_time)
            # 对于风场要素，使用 output_filename_base 作为目录名（如 wind_10m、wind_100m、wind_60m）
            # 对于其他要素，使用 element.lower() 作为目录名
            if element in ["WIND", "WIND100", "WIND60"] and "nc_mapping" in element_config:
                dir_name = element_config["nc_mapping"]["output_filename"]  # 如 wind_10m
            else:
                dir_name = element.lower()
            output_path = os.path.join(output_dir, dir_name, bjt_time_str, output_filename)

            # 4. 检查输出是否已存在（包括临时文件）
            temp_output_path = output_path + '.tmp'

            if skip_existing:
                # 如果最终文件已存在，直接返回成功
                self.logger.info(f"检查文件是否存在: {output_path}, 存在: {os.path.exists(output_path)}")
                if os.path.exists(output_path):
                    self.logger.info(f"输出文件已存在: {output_path}")
                    return True, output_path, {}

                # 如果临时文件存在，删除它（可能是之前未完成的运行）
                if os.path.exists(temp_output_path):
                    self.logger.warning(f"发现未完成的临时文件，正在删除: {temp_output_path}")
                    try:
                        os.remove(temp_output_path)
                    except Exception as e:
                        self.logger.error(f"删除临时文件失败: {str(e)}")

            # 5. 执行处理
            # 检查是否需要成对U/V分量的风场要素
            if element in ["WIND", "WIND100", "WIND60"]:
                # 风场处理需要U和V分量
                nc_files = self._find_uv_files(nc_file, element)

                if not nc_files.get("u") or not nc_files.get("v"):
                    if element == "WIND60":
                        # WIND60 依赖 100m风，如果源文件不存在则跳过并警告
                        self.logger.warning(f"未找到100m风场源文件 (100U/100V)，跳过WIND60计算")
                        return True, output_path, {}
                    else:
                        # WIND 和 WIND100 是基础要素，必须存在
                        self.logger.error(f"无法找到成对的U/V文件，请确保同时存在 U 和 V 分量文件")
                        return False, None, {}

                # 处理风场数据
                temp_success, temp_result, times_bjt, hours_new = self._process_wind_data(
                    element, nc_files, base_time, temp_output_path
                )

                if not temp_success:
                    return False, None, {}

                # 重命名临时文件为最终文件
                if os.path.exists(temp_output_path):
                    os.rename(temp_output_path, output_path)
                else:
                    self.logger.error("NetCDF文件未创建")
                    return False, None, {}
            elif element == "TEM":
                # 温度要素：处理温度数据后，自动计算相对湿度
                self.logger.info("处理温度要素，将自动计算相对湿度")

                # 1. 处理温度数据（使用临时文件路径）
                temp_success, temp_result, times_bjt, actual_output_path, hours_new = self._process_scalar_data(
                    element, nc_file, base_time, temp_output_path
                )

                if not temp_success:
                    self.logger.error("温度数据处理失败")
                    return False, None, {}

                # 2. 查找露点温度文件
                input_dir = os.path.dirname(nc_file)
                base_time_str = base_time.strftime("%Y%m%d%H")

                # 查找露点温度文件
                dpt_file_pattern = f"ECMFC1D_DPT_1_{base_time_str}*.nc"
                import glob
                dpt_files = glob.glob(os.path.join(input_dir, dpt_file_pattern))

                if dpt_files:
                    dpt_file = dpt_files[0]
                    self.logger.info(f"找到露点温度文件: {os.path.basename(dpt_file)}")

                    # 3. 处理露点温度数据
                    dpt_success, dpt_result, _, _, _ = self._process_scalar_data(
                        "DPT", dpt_file, base_time, temp_output_path + "_dpt"
                    )

                    if dpt_success and "dpt" in dpt_result:
                        # 4. 使用温度和露点温度计算相对湿度
                        self.logger.info("使用Magnus-Tetens公式计算相对湿度")
                        start_time = time.time()

                        temp_data = temp_result['temp']
                        dpt_data = dpt_result['dpt']

                        # 确保数据形状一致
                        if temp_data.shape != dpt_data.shape:
                            self.logger.warning(f"温度和露点温度数据形状不一致: temp={temp_data.shape}, dpt={dpt_data.shape}")
                            # 使用简化版本
                            rh_data = MeteorologicalCalculator.calculate_relative_humidity_from_temperature(temp_data)
                        else:
                            # 使用精确版本
                            rh_data = MeteorologicalCalculator.calculate_relative_humidity_from_temp_and_dewpoint(
                                temp_data, dpt_data
                            )

                        calc_time = time.time() - start_time
                        self.logger.info(f"相对湿度计算完成: {calc_time:.1f}秒")

                        # 5. 保存相对湿度文件
                        self.logger.info("保存相对湿度文件")
                        rh_output_filename = f"rh_0p01_1h_BJT_{Config.utc_to_bjt_str(base_time)}.nc"
                        rh_output_path = os.path.join(output_dir, "rh", bjt_time_str, rh_output_filename)

                        # 检查RH文件是否已存在
                        if skip_existing and os.path.exists(rh_output_path):
                            self.logger.info(f"相对湿度文件已存在，跳过: {rh_output_path}")
                        else:
                            # 确保输出目录存在
                            os.makedirs(os.path.dirname(rh_output_path), exist_ok=True)
                            # 写入相对湿度文件
                            with Dataset(rh_output_path, 'w') as nc:
                                # 创建维度
                                nc.createDimension('time', len(times_bjt))
                                nc.createDimension('lat', self.n_lat_dst)
                                nc.createDimension('lon', self.n_lon_dst)

                                # 坐标变量
                                time_var = nc.createVariable('time', 'i4', ('time',))
                                lat_var = nc.createVariable('lat', 'f4', ('lat',))
                                lon_var = nc.createVariable('lon', 'f4', ('lon',))

                                # 数据变量
                                rh_var = nc.createVariable('rh', 'f4', ('time', 'lat', 'lon'),
                                                           zlib=True, complevel=1)

                                # 设置数据
                                lat_var[:] = self.lat_dst
                                lat_var.units = 'degrees_north'
                                lat_var.long_name = 'latitude (south to north)'

                                lon_var[:] = self.lon_dst
                                lon_var.units = 'degrees_east'

                                time_var.units = 'hours since 1970-01-01 00:00:00'
                                time_var.calendar = 'gregorian'
                                time_var.time_zone = 'UTC+8'
                                time_var[:] = date2num(times_bjt, time_var.units, time_var.calendar)

                                rh_var[:] = rh_data
                                rh_var.units = '%'
                                rh_var.long_name = '2m relative humidity'
                                rh_var.comment = 'Calculated from temperature and dewpoint temperature using Magnus-Tetens formula'

                                # 添加全局属性
                                nc.title = 'ECMWF 2m Relative Humidity (Beijing Time)'
                                nc.source = 'ECMWF Forecast'
                                nc.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                                nc.forecast_start_time_utc = base_time.strftime("%Y-%m-%d %H:%M:%S UTC")
                                nc.forecast_start_time_bjt = Config.utc_to_bjt_str(base_time)

                            file_size_mb = os.path.getsize(rh_output_path) / 1024 / 1024
                            self.logger.info(f"相对湿度文件保存完成: {file_size_mb:.1f} MB")
                            self.logger.info(f"  数据范围: {np.nanmin(rh_data):.2f}% - {np.nanmax(rh_data):.2f}%")

                        # 清理临时露点温度文件
                        dpt_temp_path = temp_output_path + "_dpt"
                        dpt_tmp_path = temp_output_path + "_dpt.tmp"
                        if os.path.exists(dpt_temp_path):
                            os.remove(dpt_temp_path)
                        if os.path.exists(dpt_tmp_path):
                            os.remove(dpt_tmp_path)
                    else:
                        self.logger.warning("露点温度数据处理失败，跳过相对湿度计算")
                else:
                    self.logger.warning(f"未找到露点温度文件 (模式: {dpt_file_pattern})，跳过相对湿度计算")

                # 检查最终文件是否存在
                if not os.path.exists(actual_output_path):
                    self.logger.error(f"NetCDF文件未创建: {actual_output_path}")
                    return False, None, {}

            else:
                # 其他标量要素（使用临时文件路径）
                temp_success, temp_result, times_bjt, actual_output_path, hours_new = self._process_scalar_data(
                    element, nc_file, base_time, temp_output_path
                )

                if not temp_success:
                    return False, None, {}

            # 6. 保存MICAPS4格式（WIND100和WIND60不生成MICAPS4文件）
            if save_micaps4 and temp_success and element not in ["WIND100", "WIND60"]:
                # 创建要素对应的子目录
                element_output_dir = os.path.join(micaps4_output_dir, element)
                os.makedirs(element_output_dir, exist_ok=True)

                # 初始化MICAPS4文件字典
                micaps4_files = {}

                # 根据要素类型选择数据
                if element in ["WIND", "WIND100", "WIND60"]:
                    # 风场要素，从U/V分量计算风速风向
                    u_data = temp_result.get(f"u_{self._get_height_suffix(element)}")
                    v_data = temp_result.get(f"v_{self._get_height_suffix(element)}")

                    if u_data is None or v_data is None:
                        self.logger.warning(f"未找到风场U/V分量数据，无法生成MICAPS4文件")
                        temp_success = False
                    else:
                        # 计算风速风向
                        self.logger.info("计算风速风向用于MICAPS4输出")
                        data_wspd, data_wdir = MeteorologicalCalculator.calculate_wind_speed_direction(u_data, v_data)
                else:
                    # 标量要素
                    var_name = list(temp_result.keys())[0]
                    data = temp_result[var_name]

                # 为每个预报时效保存一个文件
                if element in ["WIND", "WIND100", "WIND60"] and temp_success:
                    # 风场要素，使用矢量格式写入
                    for i, (forecast_hour, wspd_slice) in enumerate(zip(hours_new, data_wspd)):
                        forecast_hour = int(forecast_hour)
                        wdir_slice = data_wdir[i]

                        # 创建MICAPS4文件名
                        micaps4_filename = MICAPS4Writer.create_micaps4_filename(
                            base_time=base_time,
                            forecast_hour=forecast_hour,
                            element=element,
                            model_name="ECMWF"
                        )

                        micaps4_path = os.path.join(element_output_dir, micaps4_filename)

                        success_write = MICAPS4Writer.write_micaps4_vector_file(
                            wspd_data=wspd_slice,
                            wdir_data=wdir_slice,
                            lats=self.lat_dst,
                            lons=self.lon_dst,
                            base_time=base_time,
                            forecast_hour=forecast_hour,
                            output_path=micaps4_path,
                            element=element,
                            model_name="ECMWF",
                            height=self._get_height_description(element)
                        )
                elif temp_success:
                    # 标量要素
                    for i, (forecast_hour, data_slice) in enumerate(zip(hours_new, data)):
                        forecast_hour = int(forecast_hour)

                        # 创建MICAPS4文件名
                        micaps4_filename = MICAPS4Writer.create_micaps4_filename(
                            base_time=base_time,
                            forecast_hour=forecast_hour,
                            element=element,
                            model_name="ECMWF"
                        )

                        micaps4_path = os.path.join(element_output_dir, micaps4_filename)

                        success_write = MICAPS4Writer.write_micaps4_scalar_file(
                            data=data_slice,
                            lats=self.lat_dst,
                            lons=self.lon_dst,
                            base_time=base_time,
                            forecast_hour=forecast_hour,
                            output_path=micaps4_path,
                            element=element,
                            model_name="ECMWF"
                        )
                    else:
                        # 标量要素
                        success_write = MICAPS4Writer.write_micaps4_scalar_file(
                            data=data_slice,
                            lats=self.lat_dst,
                            lons=self.lon_dst,
                            base_time=base_time,
                            forecast_hour=forecast_hour,
                            output_path=micaps4_path,
                            element=element,
                            model_name="ECMWF"
                        )

                    if not success_write:
                        self.logger.error(f"写入MICAPS4文件失败: {micaps4_path}")
                        return False, None, {}

                    micaps4_files[forecast_hour] = micaps4_path

            if temp_success:
                self.logger.info(f"处理完成，总耗时: {time.time() - total_start:.1f}秒")
                self.logger.info(f"NetCDF输出文件: {output_path}")

                # 返回MICAPS4文件列表
                if save_micaps4 and 'micaps4_files' in locals():
                    self.logger.info(f"MICAPS4文件数: {len(micaps4_files)}")
                    return True, output_path, micaps4_files
                else:
                    return True, output_path, {}

        except Exception as e:
            self.logger.error(f"处理要素失败: {str(e)}")
            return False, None, {}


def main_cli(args=None):
    """命令行接口"""
    parser = argparse.ArgumentParser(description='ECMWF数据处理工具 - 时空降尺度至0.01°x0.01° 1小时分辨率')

    # 基本参数
    parser.add_argument('--element', type=str, default=None,
                        choices=['WIND', 'WIND100', 'WIND60', 'GUST', 'TEM', 'PRS', 'DPT', 'MN2T6', 'MX2T6', 'VIS', 'PRE', 'TPE', 'TCC'],
                        help='要素名称（与 --input-dir 互斥，注意：RH要素在处理TEM时自动计算）')
    parser.add_argument('--nc-file', type=str, default=None,
                        help='输入的NetCDF文件路径（与 --input-dir 互斥）')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='输入目录（批量处理目录中的所有NC文件，与 --nc-file 互斥）')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录（默认: 1KM1H）')
    parser.add_argument('--base-time', type=str, default=None,
                        help='基准时间（UTC），格式: YYYYMMDDHH')
    parser.add_argument('--output-filename', type=str, default=None,
                        help='输出文件名')
    parser.add_argument('--save-micaps4', action='store_true', default=False,
                        help='保存MICAPS4格式')
    parser.add_argument('--micaps4-output-dir', type=str, default=None,
                        help='MICAPS4输出目录')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='跳过已存在的输出文件')
    parser.add_argument('--no-skip-existing', action='store_false', dest='skip_existing',
                        help='不跳过已存在的输出文件')
    parser.add_argument('--verbose', '-v', action='store_true', default=False,
                        help='详细输出')

    args = parser.parse_args(args)

    # 设置日志
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    logger = logging.getLogger(__name__)

    # 抑制警告
    warnings.filterwarnings("ignore")

    # 创建处理器
    processor = ECDataProcessor(
        logger=logger,
        save_micaps4=args.save_micaps4,
        micaps4_output_dir=args.micaps4_output_dir
    )

    # 确定输出目录
    output_dir = args.output_dir
    if output_dir is None:
        # 使用默认输出目录: 1KM1H
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '1KM1H')
        logger.info(f"使用默认输出目录: {output_dir}")

    # 处理单个文件或批量处理
    success = False
    output_path = None
    micaps4_files = {}

    # 检查是否为批量处理
    if args.input_dir is not None:
        # 批量处理模式：处理目录中的所有NC文件
        if args.nc_file is not None:
            parser.error("不能同时指定 --nc-file 和 --input-dir")
            return 1

        if args.element is not None:
            logger.info(f"批量处理模式: --element={args.element}, 将只处理匹配的要素")
        else:
            logger.info(f"批量处理模式: 处理所有NC文件")

        # 查找所有NC文件
        import glob
        nc_pattern = os.path.join(args.input_dir, "*.nc")
        nc_files = sorted(glob.glob(nc_pattern))

        if not nc_files:
            logger.error(f"未找到NC文件: {nc_pattern}")
            return 1

        logger.info(f"找到 {len(nc_files)} 个NC文件")

        # 使用Config中的映射来确定要素类型
        # 批量处理
        total_success = 0
        total_fail = 0

        for i, nc_file in enumerate(nc_files, 1):
            basename = os.path.basename(nc_file)
            logger.info(f"[{i}/{len(nc_files)}] 处理: {basename}")

            # 从文件名确定要素
            element = Config.get_element_from_filename(nc_file)

            if element is None:
                logger.warning(f"无法从文件名识别要素: {basename}, 跳过")
                total_fail += 1
                continue

            # 跳过RH要素（因为会在处理TEM时自动计算）
            if element == "RH":
                logger.debug(f"跳过RH要素（在处理TEM时自动计算）: {basename}")
                continue

            # 如果指定了要素，只处理匹配的文件
            if args.element is not None and element != args.element:
                logger.debug(f"跳过不匹配的要素: {element} != {args.element}")
                continue

            # 从文件名解析时间
            base_time = Config.parse_time_from_filename(nc_file)

            # 处理文件
            success, _, _ = processor.process_element(
                element=element,
                nc_file=nc_file,
                output_dir=output_dir,
                base_time=base_time,
                output_filename=None,
                skip_existing=args.skip_existing,
                save_micaps4=args.save_micaps4,
                micaps4_output_dir=args.micaps4_output_dir
            )

            if success:
                total_success += 1
            else:
                total_fail += 1

            # 批量处理中的内存清理
            gc.collect()

        logger.info(f"批量处理完成: 成功 {total_success}/{len(nc_files)}, 失败 {total_fail}")
        return 0 if total_fail == 0 else 1

    # 处理单个文件
    if args.nc_file is not None:
        if args.element is None:
            parser.error("使用 --nc-file 时必须指定 --element")
            return 1

        success, output_path, micaps4_files = processor.process_element(
        element=args.element,
        nc_file=args.nc_file,
        output_dir=output_dir,
        base_time=datetime.strptime(args.base_time, "%Y%m%d%H") if args.base_time else None,
        output_filename=args.output_filename,
        skip_existing=args.skip_existing
    )

    if success:
        logger.info(f"✓ 处理成功")
        return 0
    else:
        logger.error(f"✗ 处理失败")
        return 1


if __name__ == "__main__":
    sys.exit(main_cli())
