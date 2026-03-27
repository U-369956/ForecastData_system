#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECMWF 100米和60米风场数据处理系统
支持100米风场处理和60米风场计算，支持NetCDF和MICAPS4格式输出

# 处理100米风场
python EC_100m_60m_processor.py \
  --input-u-file /path/to/ECMFC1D_100U_1_2026010100_GLB_1.grib1 \
  --input-v-file /path/to/ECMFC1D_100V_1_2026010100_GLB_1.grib1 \
  --output-dir /path/to/output \
  --element WIND_100M

# 计算60米风场（从100米风场）
python EC_V2_NC_MICAPS_100mUV.py \
  --input-u-file /home/youqi/FWZX_forecast_DATA/data_demo/ECMFC1D_100U_1_2026012900_GLB_1.grib1 \
  --input-v-file /home/youqi/FWZX_forecast_DATA/data_demo/ECMFC1D_100V_1_2026012900_GLB_1.grib1 \
  --output-dir /home/youqi/FWZX_forecast_DATA/output \
  --element 'WIND_60M' 

# 批量处理目录
python EC_V3_NC_MICAPS_100mUV_20260206.py \
  --input-dir /home/youqi/FWZX_forecast_DATA/data_demo \
  --output-dir /home/youqi/FWZX_forecast_DATA/output \
  --save-micaps4 --micaps4-output-dir /home/youqi/FWZX_forecast_DATA/tonst \
  --element 'WIND_60M' 

"""

import os
import sys
import logging
import argparse
import numpy as np
from netCDF4 import Dataset, date2num
from datetime import datetime, timedelta
import pygrib
from scipy.interpolate import RegularGridInterpolator, interp1d
import warnings
import time
from typing import Tuple, List, Dict, Optional, Any
import math
import pickle
import re
from glob import glob

# ================= 配置类 =================
class Config:
    """集中管理所有配置参数"""
    # 区域范围
    REGION = {
        "lon_w": 110.0,
        "lon_e": 127.0,
        "lat_s": 34.0,
        "lat_n": 44.0
    }
    
    # 输出分辨率
    RESOLUTION = 0.01
    
    # 时区偏移（北京时）
    TIMEZONE_SHIFT = timedelta(hours=8)
    
    # 默认输出文件名格式
    OUTPUT_FILENAME_FORMAT = "{element}_0p01_1h_BJT_{time_str}.nc"
    
    # 要素配置
    ELEMENTS = {
        # 100米风场
        "WIND_100M": {
            "description": "100m wind speed and direction",
            "grib_codes": {"u": "100U", "v": "100V"},
            "requires_uv": True,
            "output_vars": ["wspd_100m", "wdir_100m"],
            "units": {"wspd_100m": "m s-1", "wdir_100m": "degree"},
            "level": "100m"
        },
        # 60米风场（从100米计算）
        "WIND_60M": {
            "description": "60m wind speed and direction (calculated from 100m wind)",
            "grib_codes": {"u": "100U", "v": "100V"},
            "requires_uv": True,
            "requires_calc": True,
            "output_vars": ["wspd_60m", "wdir_60m"],
            "units": {"wspd_60m": "m s-1", "wdir_60m": "degree"},
            "level": "60m"
        }
    }
    
    # 100米到60米的风速计算参数
    WIND_PROFILE_PARAMS = {
        "power_law_exponent": 0.143,  # 幂律指数，适用于中性大气
        "reference_height": 100.0,    # 参考高度（米）
        "target_height": 60.0,        # 目标高度（米）
        "log_law_roughness": 0.03     # 对数律粗糙度（米），适用于平坦地形
    }
    
    @classmethod
    def get_target_grid(cls) -> Tuple[np.ndarray, np.ndarray]:
        """生成目标网格 - 纬度升序（从南到北）"""
        lon_out = np.arange(cls.REGION["lon_w"], 
                           cls.REGION["lon_e"] + cls.RESOLUTION/2, 
                           cls.RESOLUTION)
        # 纬度升序（从南到北）
        lat_out = np.arange(cls.REGION["lat_s"], 
                           cls.REGION["lat_n"] + cls.RESOLUTION/2, 
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
        
        # 尝试解析格式: ECMFC1D_10U_1_2026010100_GLB_1.grib1
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
        wspd = np.sqrt(u**2 + v**2)
        
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


# ================= MICAPS4格式保存类 =================
class MICAPS4Writer:
    """MICAPS第4类数据格式写入器 - 二进制格式"""
    
    # 要素到MICAPS要素名的映射
    ELEMENT_MAP = {
        "WIND_100M": "100WIND",
        "WIND_60M": "60WIND"
    }
    
    @staticmethod
    def create_micaps4_filename(base_time: datetime, forecast_hour: int,
                               element: str = "WIND_100M", model_name: str = "ECMWF",
                               timezone_shift: timedelta = None) -> str:
        """
        创建MICAPS4格式文件名（与降水格式一致：月日时分.时效）
        
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
            MICAPS4格式文件名：月日时分.时效（3位）
        """
        if timezone_shift is None:
            timezone_shift = Config.TIMEZONE_SHIFT
        
        # 转换为本地时间
        local_time = base_time + timezone_shift
        
        # 格式: MMDDHH.时效（3位）
        return f"{local_time.strftime('%y%m%d%H')}.{forecast_hour:03d}"
    
    @staticmethod
    def write_micaps4_vector_file(wspd: np.ndarray, wdir: np.ndarray, 
                                 lats: np.ndarray, lons: np.ndarray,
                                 base_time: datetime, forecast_hour: int,
                                 output_path: str, element: str = "WIND_100M",
                                 model_name: str = "ECMWF",
                                 level: float = None,
                                 description: str = None,
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
            
            # 设置默认参数
            if level is None:
                if element == "WIND_100M":
                    level = 100.0
                elif element == "WIND_60M":
                    level = 60.0
                else:
                    level = 100.0
            
            if description is None:
                if element == "WIND_100M":
                    description = "100m wind speed(m/s) and direction(deg)"
                elif element == "WIND_60M":
                    description = "60m wind speed(m/s) and direction(deg)"
                else:
                    description = "Wind speed and direction"
            
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
                element_std = MICAPS4Writer.ELEMENT_MAP.get(element, element)
                element_bytes = element_std.encode('ascii', 'ignore')[:50]
                if len(element_bytes) < 50:
                    element_bytes += b'\x00' * (50 - len(element_bytes))
                f.write(element_bytes)
                
                # ============= 5. description =============
                if len(description) > 30:
                    description = description[:30]
                
                try:
                    desc_bytes = description.encode('gbk', 'ignore')[:30]
                except (UnicodeEncodeError, AttributeError):
                    desc_bytes = description.encode('ascii', 'ignore')[:30]
                
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
            
            file_size_kb = os.path.getsize(output_path) / 1024
            print(f"✓ MICAPS4矢量文件已生成: {output_path}")
            print(f"  数据类型: 矢量 (type=11)")
            print(f"  要素: {element_std}")
            print(f"  模式: {model_name}")
            print(f"  起报: {local_time.strftime('%Y-%m-%d %H:%M')} (UTC+8)")
            print(f"  时效: {forecast_hour:03d}小时")
            print(f"  层次: {level} hPa")
            print(f"  网格: {n_lon}x{n_lat} ({start_lon:.2f}-{end_lon:.2f}E, {start_lat:.2f}-{end_lat:.2f}N)")
            print(f"  数据点: {wspd_flat.size} 个风速 + {wdir_flat.size} 个风向")
            print(f"  文件大小: {file_size_kb:.1f} KB")
            
            return True
            
        except Exception as e:
            print(f"✗ 写入MICAPS4矢量文件失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


# ================= 日志配置 =================
def setup_logger(log_level=logging.INFO, name='ECMWF_100m_60m_Processor'):
    """配置日志系统"""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 清除现有的处理器，避免重复
    logger.handlers = []
    
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


# ================= 核心处理类 =================
class EC_100m_60m_Processor:
    """ECMWF 100米和60米风场数据处理器"""
    
    def __init__(self, logger=None, config=None, 
                 save_micaps4: bool = False,
                 micaps4_output_dir: str = None):
        """
        初始化处理器
        
        Parameters
        ----------
        logger : logging.Logger, optional
            日志记录器
        config : dict, optional
            配置参数，可覆盖默认配置
        save_micaps4 : bool, optional
            是否保存MICAPS4格式，默认为False
        micaps4_output_dir : str, optional
            MICAPS4输出目录，如为None则与NetCDF输出同目录
        """
        self.logger = logger or setup_logger()
        self.save_micaps4 = save_micaps4
        self.micaps4_output_dir = micaps4_output_dir
        
        # 更新配置（如果有）
        if config:
            for key, value in config.items():
                if hasattr(Config, key):
                    setattr(Config, key, value)
                elif key.upper() in Config.__dict__:
                    setattr(Config, key.upper(), value)
        
        # 预计算目标网格（纬度升序）
        self.lat_dst, self.lon_dst = Config.get_target_grid()
        self.n_lat_dst, self.n_lon_dst = self.lat_dst.size, self.lon_dst.size
    
    def _save_with_temp_file(self, output_path: str, save_func: callable, *args, **kwargs) -> bool:
        """
        通过临时文件保存，确保原子性操作
        
        Parameters
        ----------
        output_path : str
            最终输出路径
        save_func : callable
            保存函数，接受临时文件路径作为第一个参数
        *args, **kwargs
            传递给保存函数的其他参数
            
        Returns
        -------
        bool
            是否成功
        """
        try:
            # 创建临时文件路径
            temp_path = output_path + '.tmp'
            
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            self.logger.debug(f"开始保存到临时文件: {temp_path}")
            
            # 调用保存函数保存到临时文件
            success = save_func(temp_path, *args, **kwargs)
            
            if success:
                # 检查临时文件是否存在
                if os.path.exists(temp_path):
                    temp_size = os.path.getsize(temp_path) / 1024 / 1024
                    self.logger.debug(f"临时文件创建成功: {temp_path} ({temp_size:.1f} MB)")
                    
                    # 重命名临时文件为最终文件
                    os.rename(temp_path, output_path)
                    self.logger.info(f"✓ 文件保存完成: {output_path}")
                    return True
                else:
                    self.logger.error(f"临时文件不存在: {temp_path}")
                    return False
            else:
                # 保存失败，删除临时文件
                if os.path.exists(temp_path):
                    self.logger.debug(f"保存失败，删除临时文件: {temp_path}")
                    os.remove(temp_path)
                return False
                
        except Exception as e:
            self.logger.error(f"保存文件失败: {str(e)}")
            # 清理临时文件
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    self.logger.debug(f"异常发生时清理临时文件: {temp_path}")
                    os.remove(temp_path)
                except:
                    pass
            return False
    
    def process_element(self, element: str, input_files: Dict[str, str], 
                   output_dir: str = None, base_time: datetime = None,
                   output_filename: str = None, skip_existing: bool = True,
                   save_micaps4: bool = None, micaps4_output_dir: str = None) -> Tuple[bool, str, Dict]:
        """
        处理单个要素数据
        
        Parameters
        ----------
        element : str
            要素名称（WIND_100M, WIND_60M）
        input_files : dict
            输入文件路径字典
        output_dir : str, optional
            输出目录
        base_time : datetime, optional
            基准时间（UTC）
        output_filename : str, optional
            输出文件名
        skip_existing : bool, optional
            是否跳过已存在的输出文件
        save_micaps4 : bool, optional
            是否保存MICAPS4格式
        micaps4_output_dir : str, optional
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
            
            # 2. 检查输入文件
            for file_key, file_path in input_files.items():
                if not os.path.exists(file_path):
                    self.logger.error(f"文件不存在: {file_path} (key: {file_key})")
                    return False, None, {}
            
            # 3. 确定基准时间
            if base_time is None:
                # 从第一个文件解析时间
                first_file = list(input_files.values())[0]
                base_time = Config.parse_time_from_filename(first_file)
            
            if base_time is None:
                self.logger.error("无法从文件名解析时间，请提供base_time参数")
                return False, None, {}
            
            # 4. 确定输出路径
            if output_dir is None:
                # 使用第一个文件的目录
                first_file = list(input_files.values())[0]
                output_dir = os.path.dirname(first_file)
            
            if output_filename is None:
                bjt_time_str = Config.utc_to_bjt_str(base_time)
                output_filename = Config.OUTPUT_FILENAME_FORMAT.format(
                    element=element.lower(), time_str=bjt_time_str
                )
            
            output_path = os.path.join(output_dir, output_filename)
            
            # 5. 检查输出是否已存在（包括临时文件）
            temp_output_path = output_path + '.tmp'
            
            if skip_existing:
                # 如果最终文件已存在，直接返回成功
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
            
            # 6. 执行处理
            if element == "WIND_60M":
                # 60米风场需要从100米风场计算
                success, result_data, times_bjt = self._process_60m_wind_data(
                    input_files, base_time, output_path
                )
            else:
                # 100米风场直接处理
                success, result_data, times_bjt = self._process_100m_wind_data(
                    input_files, base_time, output_path
                )
            
            # 7. 保存MICAPS4格式
            if success and save_micaps4 and result_data is not None:
                micaps4_files = self._save_micaps4_files(
                    element, result_data, base_time, micaps4_output_dir
                )
            
            if success:
                total_time = time.time() - total_start
                self.logger.info(f"处理完成，总耗时: {total_time:.1f}秒")
                self.logger.info(f"NetCDF输出文件: {output_path}")
                
                if micaps4_files:
                    micaps4_count = sum(len(files) for files in micaps4_files.values())
                    self.logger.info(f"MICAPS4文件: {micaps4_count} 个")
            
            return success, output_path, micaps4_files
            
        except Exception as e:
            self.logger.error(f"处理失败: {str(e)}", exc_info=True)
            import traceback
            traceback.print_exc()
            return False, output_path, micaps4_files
    
    def _process_100m_wind_data(self, input_files: Dict[str, str],
                               base_time_utc: datetime, output_path: str) -> Tuple[bool, Dict[str, np.ndarray], List[datetime]]:
        """处理100米风场数据（需要U/V分量）"""
        try:
            self.logger.info("处理100米风场数据（U/V分量）")
            
            # 检查输入文件
            u_file = input_files.get("u")
            v_file = input_files.get("v")
            
            if not u_file or not v_file:
                self.logger.error("100米风场处理需要U和V分量文件")
                return False, None, None
            
            # 1. 读取数据
            start_time = time.time()
            with pygrib.open(u_file) as grbs_u, pygrib.open(v_file) as grbs_v:
                # 获取所有时次
                steps_u = sorted({msg.step for msg in grbs_u})
                steps_v = sorted({msg.step for msg in grbs_v})
                common_steps = sorted(set(steps_u) & set(steps_v))
                
                if not common_steps:
                    self.logger.error("没有共同的预报时次")
                    return False, None, None
                
                # 根据时效分两段
                steps_3h = [step for step in common_steps if step <= 72]
                steps_6h = [step for step in common_steps if step > 72 and step <= 240]
                
                if not steps_3h and not steps_6h:
                    self.logger.error("没有有效的预报时次")
                    return False, None, None
                
                # 读取第一个消息获取网格信息
                msg_u = grbs_u.select(step=common_steps[0])[0]
                data, lats, lons = msg_u.data(
                    lat1=Config.REGION["lat_s"],
                    lat2=Config.REGION["lat_n"],
                    lon1=Config.REGION["lon_w"],
                    lon2=Config.REGION["lon_e"]
                )
                
                # 获取源网格
                lat_src = lats[:, 0]
                lon_src = lons[0, :]
                
                if lat_src[-1] < lat_src[0]:
                    lat_src = lat_src[::-1]
                
                # 初始化存储
                all_u_data = []
                all_v_data = []
                all_steps = []
                
                # 读取3小时间隔数据
                if steps_3h:
                    u_3h, v_3h = self._read_uv_data(grbs_u, grbs_v, steps_3h, lat_src)
                    all_u_data.extend(u_3h)
                    all_v_data.extend(v_3h)
                    all_steps.extend(steps_3h)
                
                # 读取6小时间隔数据
                if steps_6h:
                    u_6h, v_6h = self._read_uv_data(grbs_u, grbs_v, steps_6h, lat_src)
                    all_u_data.extend(u_6h)
                    all_v_data.extend(v_6h)
                    all_steps.extend(steps_6h)
                
                u_cube = np.array(all_u_data, dtype=np.float32)
                v_cube = np.array(all_v_data, dtype=np.float32)
            
            read_time = time.time() - start_time
            self.logger.info(f"数据读取完成: {read_time:.1f}秒, 总时次: {len(all_steps)}")
            
            # 2. 空间插值
            start_time = time.time()
            
            lon2d, lat2d = np.meshgrid(self.lon_dst, self.lat_dst)
            points = np.column_stack([lat2d.ravel(), lon2d.ravel()])
            
            interp_func = RegularGridInterpolator(
                (lat_src, lon_src),
                u_cube[0],
                bounds_error=False,
                fill_value=np.nan
            )
            
            u_interp = np.empty((len(all_steps), self.n_lat_dst, self.n_lon_dst), dtype=np.float32)
            v_interp = np.empty_like(u_interp)
            
            for i in range(len(all_steps)):
                interp_func.values = u_cube[i]
                u_interp[i] = interp_func(points).reshape(self.n_lat_dst, self.n_lon_dst)
                interp_func.values = v_cube[i]
                v_interp[i] = interp_func(points).reshape(self.n_lat_dst, self.n_lon_dst)
            
            interp_time = time.time() - start_time
            self.logger.info(f"空间插值完成: {interp_time:.1f}秒")

            # 3. 时间插值（分段处理）- 先对u、v进行时间插值
            start_time = time.time()

            # 分段处理
            u_1h_segments = []
            v_1h_segments = []
            hour_segments = []

            # 第一段: 0-72小时
            if steps_3h:
                indices_3h = [i for i, step in enumerate(all_steps) if step in steps_3h]
                u_3h = u_interp[indices_3h]
                v_3h = v_interp[indices_3h]

                hours_3h = steps_3h
                hours_new_3h = np.arange(0, 72 + 1, 1.0)
                hours_new_3h = hours_new_3h[hours_new_3h <= 72]

                f_u_3h = interp1d(hours_3h, u_3h, axis=0, kind='linear', fill_value='extrapolate')
                f_v_3h = interp1d(hours_3h, v_3h, axis=0, kind='linear', fill_value='extrapolate')

                u_1h_3h = f_u_3h(hours_new_3h)
                v_1h_3h = f_v_3h(hours_new_3h)

                u_1h_segments.append(u_1h_3h)
                v_1h_segments.append(v_1h_3h)
                hour_segments.append(hours_new_3h)

                self.logger.info(f"  0-72小时: {len(hours_3h)}个原始时次 → {len(hours_new_3h)}个1小时间隔时次")

            # 第二段: 72-240小时
            if steps_6h:
                indices_6h = [i for i, step in enumerate(all_steps) if step in steps_6h]
                u_6h = u_interp[indices_6h]
                v_6h = v_interp[indices_6h]

                hours_6h = steps_6h
                max_forecast_hour = max(steps_6h)
                start_hour_6h = 73
                end_hour_6h = min(max_forecast_hour, 240)

                if start_hour_6h <= end_hour_6h:
                    hours_new_6h = np.arange(start_hour_6h, end_hour_6h + 1, 1.0)

                    f_u_6h = interp1d(hours_6h, u_6h, axis=0, kind='linear', fill_value='extrapolate')
                    f_v_6h = interp1d(hours_6h, v_6h, axis=0, kind='linear', fill_value='extrapolate')

                    u_1h_6h = f_u_6h(hours_new_6h)
                    v_1h_6h = f_v_6h(hours_new_6h)

                    u_1h_segments.append(u_1h_6h)
                    v_1h_segments.append(v_1h_6h)
                    hour_segments.append(hours_new_6h)

                    self.logger.info(f"  73-{end_hour_6h}小时: {len(hours_6h)}个原始时次 → {len(hours_new_6h)}个1小时间隔时次")

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

            # 4. 用插值后的U、V计算风速风向
            start_time = time.time()
            wspd_1h, wdir_1h = MeteorologicalCalculator.calculate_wind_speed_direction(u_1h, v_1h)
            calc_time = time.time() - start_time
            self.logger.info(f"风速风向计算完成: {calc_time:.1f}秒")

            times_bjt = [base_time_utc + timedelta(hours=float(h)) + Config.TIMEZONE_SHIFT
                        for h in hours_new]

            # 5. 写入NetCDF（使用临时文件）
            start_time = time.time()
            
            # 定义内部函数用于保存NetCDF
            def save_netcdf_func(temp_output_path):
                with Dataset(temp_output_path, 'w') as nc:
                    # 创建维度
                    nc.createDimension('time', len(times_bjt))
                    nc.createDimension('lat', self.n_lat_dst)
                    nc.createDimension('lon', self.n_lon_dst)
                    
                    # 坐标变量
                    time_var = nc.createVariable('time', 'i4', ('time',))
                    lat_var = nc.createVariable('lat', 'f4', ('lat',))
                    lon_var = nc.createVariable('lon', 'f4', ('lon',))
                    
                    # 数据变量
                    wspd_var = nc.createVariable('wspd_100m', 'f4', ('time', 'lat', 'lon'),
                                                 zlib=True, complevel=1)
                    wdir_var = nc.createVariable('wdir_100m', 'f4', ('time', 'lat', 'lon'),
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
                    
                    wspd_var[:] = wspd_1h
                    wspd_var.units = 'm s-1'
                    wspd_var.long_name = '100 meter wind speed'
                    
                    wdir_var[:] = wdir_1h
                    wdir_var.units = 'degree'
                    wdir_var.long_name = '100 meter wind direction'
                    wdir_var.comment = '0° indicates wind from north, 90° from east'
                    
                    # 添加全局属性
                    nc.title = 'ECMWF 100m Wind Speed and Direction (Beijing Time)'
                    nc.source = 'ECMWF Forecast'
                    nc.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                    nc.forecast_start_time_utc = base_time_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
                    nc.forecast_start_time_bjt = Config.utc_to_bjt_str(base_time_utc)
                    nc.latitude_order = 'south to north (ascending)'
                    nc.time_interpolation = '分段线性插值: 0-72小时(3h→1h), 73-240小时(6h→1h)'
                
                return True
            
            # 通过临时文件保存
            success = self._save_with_temp_file(output_path, save_netcdf_func)
            
            if success:
                write_time = time.time() - start_time
                file_size_mb = os.path.getsize(output_path) / 1024 / 1024
                self.logger.info(f"NetCDF文件写入完成: {write_time:.1f}秒, 大小: {file_size_mb:.1f} MB")
                
                # 返回数据
                result_data = {
                    "u_100m": u_interp, 
                    "v_100m": v_interp, 
                    "wspd_100m": wspd_1h, 
                    "wdir_100m": wdir_1h
                }
                return True, result_data, times_bjt
            else:
                self.logger.error("NetCDF文件保存失败")
                return False, None, None
            
        except Exception as e:
            self.logger.error(f"处理100米风场数据失败: {str(e)}")
            return False, None, None
    
    def _process_60m_wind_data(self, input_files: Dict[str, str],
                              base_time_utc: datetime, output_path: str) -> Tuple[bool, Dict[str, np.ndarray], List[datetime]]:
        """处理60米风场数据（从100米风场计算）"""
        try:
            self.logger.info("处理60米风场数据（从100米风场计算）")
            
            # 检查输入文件
            u_file = input_files.get("u")
            v_file = input_files.get("v")
            
            if not u_file or not v_file:
                self.logger.error("60米风场处理需要100米U和V分量文件")
                return False, None, None
            
            # 1. 读取100米风场数据
            start_time = time.time()
            with pygrib.open(u_file) as grbs_u, pygrib.open(v_file) as grbs_v:
                # 获取所有时次
                steps_u = sorted({msg.step for msg in grbs_u})
                steps_v = sorted({msg.step for msg in grbs_v})
                common_steps = sorted(set(steps_u) & set(steps_v))
                
                if not common_steps:
                    self.logger.error("没有共同的预报时次")
                    return False, None, None
                
                # 根据时效分两段
                steps_3h = [step for step in common_steps if step <= 72]
                steps_6h = [step for step in common_steps if step > 72 and step <= 240]
                
                if not steps_3h and not steps_6h:
                    self.logger.error("没有有效的预报时次")
                    return False, None, None
                
                # 读取第一个消息获取网格信息
                msg_u = grbs_u.select(step=common_steps[0])[0]
                data, lats, lons = msg_u.data(
                    lat1=Config.REGION["lat_s"],
                    lat2=Config.REGION["lat_n"],
                    lon1=Config.REGION["lon_w"],
                    lon2=Config.REGION["lon_e"]
                )
                
                # 获取源网格
                lat_src = lats[:, 0]
                lon_src = lons[0, :]
                
                if lat_src[-1] < lat_src[0]:
                    lat_src = lat_src[::-1]
                
                # 初始化存储
                all_u_data = []
                all_v_data = []
                all_steps = []
                
                # 读取3小时间隔数据
                if steps_3h:
                    u_3h, v_3h = self._read_uv_data(grbs_u, grbs_v, steps_3h, lat_src)
                    all_u_data.extend(u_3h)
                    all_v_data.extend(v_3h)
                    all_steps.extend(steps_3h)
                
                # 读取6小时间隔数据
                if steps_6h:
                    u_6h, v_6h = self._read_uv_data(grbs_u, grbs_v, steps_6h, lat_src)
                    all_u_data.extend(u_6h)
                    all_v_data.extend(v_6h)
                    all_steps.extend(steps_6h)
                
                u_100m_cube = np.array(all_u_data, dtype=np.float32)
                v_100m_cube = np.array(all_v_data, dtype=np.float32)
            
            read_time = time.time() - start_time
            self.logger.info(f"100米风场数据读取完成: {read_time:.1f}秒, 总时次: {len(all_steps)}")
            
            # 2. 空间插值
            start_time = time.time()
            
            lon2d, lat2d = np.meshgrid(self.lon_dst, self.lat_dst)
            points = np.column_stack([lat2d.ravel(), lon2d.ravel()])
            
            interp_func = RegularGridInterpolator(
                (lat_src, lon_src),
                u_100m_cube[0],
                bounds_error=False,
                fill_value=np.nan
            )
            
            u_100m_interp = np.empty((len(all_steps), self.n_lat_dst, self.n_lon_dst), dtype=np.float32)
            v_100m_interp = np.empty_like(u_100m_interp)
            
            for i in range(len(all_steps)):
                interp_func.values = u_100m_cube[i]
                u_100m_interp[i] = interp_func(points).reshape(self.n_lat_dst, self.n_lon_dst)
                interp_func.values = v_100m_cube[i]
                v_100m_interp[i] = interp_func(points).reshape(self.n_lat_dst, self.n_lon_dst)
            
            interp_time = time.time() - start_time
            self.logger.info(f"空间插值完成: {interp_time:.1f}秒")
            
            # 3. 计算60米风场
            start_time = time.time()
            self.logger.info("从100米风场计算60米风场...")

            # 使用幂律方法计算60米风场
            u_60m_interp = np.empty_like(u_100m_interp)
            v_60m_interp = np.empty_like(v_100m_interp)

            for i in range(len(all_steps)):
                u_60m_interp[i], v_60m_interp[i] = MeteorologicalCalculator.calculate_60m_wind_from_100m(
                    u_100m_interp[i], v_100m_interp[i], method='power_law'
                )

            calc_time = time.time() - start_time
            self.logger.info(f"60米风场计算完成: {calc_time:.1f}秒")

            # 4. 时间插值（分段处理）- 先对u_60m、v_60m进行时间插值
            start_time = time.time()

            # 分段处理
            u_1h_segments = []
            v_1h_segments = []
            hour_segments = []

            # 第一段: 0-72小时
            if steps_3h:
                indices_3h = [i for i, step in enumerate(all_steps) if step in steps_3h]
                u_3h = u_60m_interp[indices_3h]
                v_3h = v_60m_interp[indices_3h]

                hours_3h = steps_3h
                hours_new_3h = np.arange(0, 72 + 1, 1.0)
                hours_new_3h = hours_new_3h[hours_new_3h <= 72]

                f_u_3h = interp1d(hours_3h, u_3h, axis=0, kind='linear', fill_value='extrapolate')
                f_v_3h = interp1d(hours_3h, v_3h, axis=0, kind='linear', fill_value='extrapolate')

                u_1h_3h = f_u_3h(hours_new_3h)
                v_1h_3h = f_v_3h(hours_new_3h)

                u_1h_segments.append(u_1h_3h)
                v_1h_segments.append(v_1h_3h)
                hour_segments.append(hours_new_3h)

                self.logger.info(f"  0-72小时: {len(hours_3h)}个原始时次 → {len(hours_new_3h)}个1小时间隔时次")

            # 第二段: 72-240小时
            if steps_6h:
                indices_6h = [i for i, step in enumerate(all_steps) if step in steps_6h]
                u_6h = u_60m_interp[indices_6h]
                v_6h = v_60m_interp[indices_6h]

                hours_6h = steps_6h
                max_forecast_hour = max(steps_6h)
                start_hour_6h = 73
                end_hour_6h = min(max_forecast_hour, 240)

                if start_hour_6h <= end_hour_6h:
                    hours_new_6h = np.arange(start_hour_6h, end_hour_6h + 1, 1.0)

                    f_u_6h = interp1d(hours_6h, u_6h, axis=0, kind='linear', fill_value='extrapolate')
                    f_v_6h = interp1d(hours_6h, v_6h, axis=0, kind='linear', fill_value='extrapolate')

                    u_1h_6h = f_u_6h(hours_new_6h)
                    v_1h_6h = f_v_6h(hours_new_6h)

                    u_1h_segments.append(u_1h_6h)
                    v_1h_segments.append(v_1h_6h)
                    hour_segments.append(hours_new_6h)

                    self.logger.info(f"  73-{end_hour_6h}小时: {len(hours_6h)}个原始时次 → {len(hours_new_6h)}个1小时间隔时次")

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

            # 5. 用插值后的U、V计算风速风向
            start_time = time.time()
            wspd_1h, wdir_1h = MeteorologicalCalculator.calculate_wind_speed_direction(u_1h, v_1h)
            calc_time = time.time() - start_time
            self.logger.info(f"风速风向计算完成: {calc_time:.1f}秒")

            times_bjt = [base_time_utc + timedelta(hours=float(h)) + Config.TIMEZONE_SHIFT
                        for h in hours_new]

            # 6. 写入NetCDF（使用临时文件）
            start_time = time.time()
            
            # 定义内部函数用于保存NetCDF
            def save_netcdf_func(temp_output_path):
                with Dataset(temp_output_path, 'w') as nc:
                    # 创建维度
                    nc.createDimension('time', len(times_bjt))
                    nc.createDimension('lat', self.n_lat_dst)
                    nc.createDimension('lon', self.n_lon_dst)
                    
                    # 坐标变量
                    time_var = nc.createVariable('time', 'i4', ('time',))
                    lat_var = nc.createVariable('lat', 'f4', ('lat',))
                    lon_var = nc.createVariable('lon', 'f4', ('lon',))
                    
                    # 数据变量
                    wspd_var = nc.createVariable('wspd_60m', 'f4', ('time', 'lat', 'lon'),
                                                 zlib=True, complevel=1)
                    wdir_var = nc.createVariable('wdir_60m', 'f4', ('time', 'lat', 'lon'),
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
                    
                    wspd_var[:] = wspd_1h
                    wspd_var.units = 'm s-1'
                    wspd_var.long_name = '60 meter wind speed'
                    wspd_var.comment = 'Calculated from 100m wind using power law method'
                    
                    wdir_var[:] = wdir_1h
                    wdir_var.units = 'degree'
                    wdir_var.long_name = '60 meter wind direction'
                    wdir_var.comment = '0° indicates wind from north, 90° from east'
                    
                    # 添加全局属性
                    nc.title = 'ECMWF 60m Wind Speed and Direction (Beijing Time)'
                    nc.source = 'ECMWF Forecast'
                    nc.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                    nc.forecast_start_time_utc = base_time_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
                    nc.forecast_start_time_bjt = Config.utc_to_bjt_str(base_time_utc)
                    nc.latitude_order = 'south to north (ascending)'
                    nc.time_interpolation = '分段线性插值: 0-72小时(3h→1h), 73-240小时(6h→1h)'
                    nc.calculation_method = 'Calculated from 100m wind using power law method'
                    nc.power_law_exponent = str(Config.WIND_PROFILE_PARAMS["power_law_exponent"])
                
                return True
            
            # 通过临时文件保存
            success = self._save_with_temp_file(output_path, save_netcdf_func)
            
            if success:
                write_time = time.time() - start_time
                file_size_mb = os.path.getsize(output_path) / 1024 / 1024
                self.logger.info(f"NetCDF文件写入完成: {write_time:.1f}秒, 大小: {file_size_mb:.1f} MB")
                
                # 返回数据
                result_data = {
                    "wspd_60m": wspd_1h, 
                    "wdir_60m": wdir_1h
                }
                return True, result_data, times_bjt
            else:
                self.logger.error("NetCDF文件保存失败")
                return False, None, None
            
        except Exception as e:
            self.logger.error(f"处理60米风场数据失败: {str(e)}")
            return False, None, None
    
    def _read_uv_data(self, grbs_u, grbs_v, steps, lat_src):
        """读取U/V分量数据"""
        u_data = []
        v_data = []
        
        for step in steps:
            # 读取U分量
            msg = grbs_u.select(step=step)[0]
            data, lats, lons = msg.data(
                lat1=Config.REGION["lat_s"],
                lat2=Config.REGION["lat_n"],
                lon1=Config.REGION["lon_w"],
                lon2=Config.REGION["lon_e"]
            )
            
            if lat_src[-1] > lat_src[0]:  # 源纬度升序
                if lats[-1, 0] < lats[0, 0]:  # 数据纬度降序
                    data = data[::-1, :]
            else:  # 源纬度降序
                if lats[-1, 0] > lats[0, 0]:  # 数据纬度升序
                    data = data[::-1, :]
            
            u_data.append(data)
            
            # 读取V分量
            msg = grbs_v.select(step=step)[0]
            data, lats, lons = msg.data(
                lat1=Config.REGION["lat_s"],
                lat2=Config.REGION["lat_n"],
                lon1=Config.REGION["lon_w"],
                lon2=Config.REGION["lon_e"]
            )
            
            if lat_src[-1] > lat_src[0]:
                if lats[-1, 0] < lats[0, 0]:
                    data = data[::-1, :]
            else:
                if lats[-1, 0] > lats[0, 0]:
                    data = data[::-1, :]
            
            v_data.append(data)
        
        return u_data, v_data
    
    def _save_micaps4_files(self, element: str, result_data: Dict[str, np.ndarray],
                           base_time_utc: datetime, output_dir: str) -> Dict[str, List[str]]:
        """
        保存MICAPS4格式文件（按要素创建子文件夹）
        
        Parameters
        ----------
        element : str
            要素名称
        result_data : dict
            处理结果数据
        base_time_utc : datetime
            UTC基准时间
        output_dir : str
            输出目录
            
        Returns
        -------
        dict
            文件路径字典
        """
        if not output_dir:
            self.logger.warning("未指定MICAPS4输出目录，跳过保存")
            return {}
        
        self.logger.info(f"开始保存MICAPS4格式文件: {element}")
        
        # 获取要素对应的MICAPS要素名
        element_name = MICAPS4Writer.ELEMENT_MAP.get(element, element)
        
        # 创建要素子文件夹
        element_dir = os.path.join(output_dir, element_name)
        os.makedirs(element_dir, exist_ok=True)
        self.logger.info(f"MICAPS4文件将保存到: {element_dir}")
        
        micaps4_files = {"vector": []}
        
        start_time = time.time()
        
        if element in ["WIND_100M", "WIND_60M"]:
            # 风场数据（矢量格式）
            if element == "WIND_100M":
                wspd_data = result_data.get("wspd_100m")
                wdir_data = result_data.get("wdir_100m")
                level = 100.0
                description = "100m wind speed and direction"
            else:  # WIND_60M
                wspd_data = result_data.get("wspd_60m")
                wdir_data = result_data.get("wdir_60m")
                level = 60.0
                description = "60m wind speed and direction (calculated from 100m)"
            
            if wspd_data is not None and wdir_data is not None:
                for i in range(wspd_data.shape[0]):
                    forecast_hour = i
                    
                    # 生成文件名：月日时分.时效
                    filename = MICAPS4Writer.create_micaps4_filename(
                        base_time=base_time_utc,
                        forecast_hour=forecast_hour,
                        element=element,
                        model_name="ECMWF"
                    )
                    output_path = os.path.join(element_dir, filename)
                    
                    # 跳过全为NaN的数据
                    if np.all(np.isnan(wspd_data[i])) or np.all(np.isnan(wdir_data[i])):
                        continue
                    
                    success = MICAPS4Writer.write_micaps4_vector_file(
                        wspd=wspd_data[i], wdir=wdir_data[i],
                        lats=self.lat_dst, lons=self.lon_dst,
                        base_time=base_time_utc, forecast_hour=forecast_hour,
                        output_path=output_path, element=element, model_name="ECMWF",
                        level=level, description=description
                    )
                    
                    if success:
                        micaps4_files["vector"].append(output_path)
                    
                    # 每处理100个时次打印一次进度
                    if (i + 1) % 100 == 0 or i == wspd_data.shape[0] - 1:
                        self.logger.info(f"  进度: {i+1}/{wspd_data.shape[0]} 个时次")
        
        total_time = time.time() - start_time
        total_files = len(micaps4_files["vector"])
        self.logger.info(f"MICAPS4文件保存完成: 共{total_files}个文件, 耗时: {total_time:.1f}秒")
        self.logger.info(f"文件保存目录: {element_dir}")
        
        return micaps4_files
    
    def batch_process(self, element: str, file_sets: List[Dict[str, str]], 
                     output_dir: str = None, base_times: List[datetime] = None,
                     skip_existing: bool = True, save_micaps4: bool = None,
                     micaps4_output_dir: str = None) -> Dict[str, int]:
        """
        批量处理多个文件集
        
        Parameters
        ----------
        element : str
            要素名称
        file_sets : list of dict
            文件集列表，每个元素为文件路径字典
        output_dir : str, optional
            输出目录
        base_times : list of datetime, optional
            基准时间列表
        skip_existing : bool, optional
            是否跳过已存在的输出文件
        save_micaps4 : bool, optional
            是否保存MICAPS4格式
        micaps4_output_dir : str, optional
            MICAPS4输出目录
            
        Returns
        -------
        dict
            处理统计信息
        """
        if save_micaps4 is None:
            save_micaps4 = self.save_micaps4
        
        stats = {'total': 0, 'success': 0, 'failed': 0, 'micaps4_files': 0}
        
        total_start = time.time()
        self.logger.info(f"开始批量处理 {len(file_sets)} 个文件集 - 要素: {element}")
        self.logger.info(f"保存MICAPS4格式: {'是' if save_micaps4 else '否'}")
        
        for i, file_set in enumerate(file_sets):
            stats['total'] += 1
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"处理文件集 {i+1}/{len(file_sets)}")
            
            # 确定基准时间
            base_time = None
            if base_times and i < len(base_times):
                base_time = base_times[i]
            
            # 处理文件
            success, output_path, micaps4_files = self.process_element(
                element=element,
                input_files=file_set,
                output_dir=output_dir,
                base_time=base_time,
                skip_existing=skip_existing,
                save_micaps4=save_micaps4,
                micaps4_output_dir=micaps4_output_dir
            )
            
            if success:
                stats['success'] += 1
                if micaps4_files:
                    micaps4_count = sum(len(files) for files in micaps4_files.values())
                    stats['micaps4_files'] += micaps4_count
            else:
                stats['failed'] += 1
            
            time.sleep(0.1)
        
        total_time = time.time() - total_start
        self.logger.info(f"\n{'='*80}")
        self.logger.info("批量处理完成！")
        self.logger.info(f"总文件集: {stats['total']}")
        self.logger.info(f"成功: {stats['success']}")
        self.logger.info(f"失败: {stats['failed']}")
        if save_micaps4:
            self.logger.info(f"MICAPS4文件: {stats['micaps4_files']} 个")
        self.logger.info(f"总耗时: {total_time:.1f}秒")
        if stats['total'] > 0:
            self.logger.info(f"平均时间: {total_time/stats['total']:.1f}秒/文件集")
        self.logger.info(f"{'='*80}")
        
        return stats


# ================= 辅助函数 =================
def create_file_sets_from_directory(input_dir: str, element: str) -> List[Dict[str, str]]:
    """
    从目录创建文件集
    
    Parameters
    ----------
    input_dir : str
        输入目录
    element : str
        要素名称
        
    Returns
    -------
    list
        文件集列表
    """
    file_sets = []
    
    # 获取要素配置
    if element not in Config.ELEMENTS:
        print(f"错误: 不支持的元素 {element}")
        return []
    
    element_config = Config.ELEMENTS[element]
    
    # 风场需要U/V文件对
    # 查找所有U分量文件
    u_files = []
    for pattern in ["*100U*.grib*", "*100U*.grb*", "*100u*.grib*", "*100u*.grb*"]:
        u_files.extend(glob(os.path.join(input_dir, pattern)))
    u_files = sorted(set(u_files))
    
    for u_file in u_files:
        # 尝试多种方式找到对应的V文件
        v_file_candidates = []
        
        # 方式1: 直接替换100U为100V
        v_file_candidates.append(u_file.replace("100U", "100V"))
        v_file_candidates.append(u_file.replace("100u", "100v"))
        
        # 方式2: 使用正则表达式替换
        base_name = os.path.basename(u_file)
        v_base_name = re.sub(r'100[Uu]', '100V', base_name)
        v_file_candidates.append(os.path.join(input_dir, v_base_name))
        
        # 方式3: 查找时间戳匹配的V文件
        # 从文件名中提取时间戳
        time_match = re.search(r'\d{10}', base_name)
        if time_match:
            time_str = time_match.group(0)
            # 查找包含相同时间戳的V文件
            v_pattern = f"*100V*{time_str}*.grib*"
            v_files = glob(os.path.join(input_dir, v_pattern))
            if v_files:
                v_file_candidates.extend(v_files)
        
        # 查找存在的V文件
        v_file = None
        for candidate in v_file_candidates:
            if os.path.exists(candidate):
                v_file = candidate
                break
        
        if v_file and os.path.exists(v_file):
            file_sets.append({"u": u_file, "v": v_file})
            print(f"找到100米风场文件对: U={os.path.basename(u_file)}, V={os.path.basename(v_file)}")
        else:
            print(f"警告: 找不到对应的100V文件: {os.path.basename(u_file)}")
    
    print(f"为要素 {element} 找到 {len(file_sets)} 个文件集")
    return file_sets


# ================= 命令行接口 =================
def main_cli(args=None):
    """命令行接口主函数"""
    parser = argparse.ArgumentParser(description='ECMWF 100米和60米风场数据处理器')
    
    # 基本参数
    parser.add_argument('--elements', type=str, default=None,
                       help='要素名称列表，用逗号分隔，如: WIND_100M,WIND_60M')
    
    parser.add_argument('--element', type=str, default=None,
                       choices=['WIND_100M', 'WIND_60M'],
                       help='单个要素名称（与--elements互斥）')
    
    # 输入方式1: 单个文件/文件集
    parser.add_argument('--input-u-file', type=str, default=None,
                       help='U分量文件路径（100米风场）')
    
    parser.add_argument('--input-v-file', type=str, default=None,
                       help='V分量文件路径（100米风场）')
    
    # 输入方式2: 目录批量处理
    parser.add_argument('--input-dir', type=str, default=None,
                       help='输入目录（包含要素文件）')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default=None,
                       help='NetCDF输出目录')
    
    parser.add_argument('--base-time', type=str, default=None,
                       help='基准时间（UTC），格式: YYYYMMDDHH')
    
    # MICAPS4参数
    parser.add_argument('--save-micaps4', action='store_true', default=False,
                       help='保存MICAPS4格式文件')
    
    parser.add_argument('--micaps4-output-dir', type=str, default=None,
                       help='MICAPS4输出目录')
    
    # 处理选项
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='跳过已存在的输出文件')
    
    parser.add_argument('--no-skip-existing', action='store_false', dest='skip_existing',
                       help='不跳过已存在的输出文件')
    
    # 配置参数
    parser.add_argument('--lon-west', type=float, default=110.0,
                       help='区域西边界经度')
    
    parser.add_argument('--lon-east', type=float, default=127.0,
                       help='区域东边界经度')
    
    parser.add_argument('--lat-south', type=float, default=34.0,
                       help='区域南边界纬度')
    
    parser.add_argument('--lat-north', type=float, default=44.0,
                       help='区域北边界纬度')
    
    parser.add_argument('--resolution', type=float, default=0.01,
                       help='输出分辨率（度）')
    
    # 风廓线参数
    parser.add_argument('--power-law-exponent', type=float, default=0.143,
                       help='幂律指数（用于从100米计算60米风速）')
    
    parser.add_argument('--calculation-method', type=str, default='power_law',
                       choices=['power_law', 'log_law'],
                       help='60米风场计算方法')
    
    args = parser.parse_args(args)
    
    # 检查要素参数
    if args.elements is None and args.element is None:
        print("错误: 必须提供--element或--elements参数")
        parser.print_help()
        return 1
    
    if args.elements is not None and args.element is not None:
        print("错误: --element和--elements参数不能同时使用")
        return 1
    
    # 解析要素列表
    if args.elements:
        # 解析逗号分隔的要素列表
        elements = [elem.strip().upper() for elem in args.elements.split(',')]
        # 检查要素是否有效
        valid_elements = ['WIND_100M', 'WIND_60M']
        for elem in elements:
            if elem not in valid_elements:
                print(f"错误: 无效的要素名称 '{elem}'，有效要素: {valid_elements}")
                return 1
    else:
        # 单个要素
        elements = [args.element]
    
    # 检查输入参数
    if args.input_dir is None and (args.input_u_file is None or args.input_v_file is None):
        print("错误: 必须提供输入目录或U/V分量文件")
        parser.print_help()
        return 1
    
    # 解析基准时间
    base_time = None
    if args.base_time:
        try:
            base_time = datetime.strptime(args.base_time, "%Y%m%d%H")
        except ValueError:
            print(f"错误: 无法解析基准时间 {args.base_time}")
            return 1
    
    # 更新配置
    config = {
        "REGION": {
            "lon_w": args.lon_west,
            "lon_e": args.lon_east,
            "lat_s": args.lat_south,
            "lat_n": args.lat_north
        },
        "RESOLUTION": args.resolution,
        "WIND_PROFILE_PARAMS": {
            "power_law_exponent": args.power_law_exponent,
            "reference_height": 100.0,
            "target_height": 60.0,
            "log_law_roughness": 0.03
        }
    }
    
    # 设置日志
    logger = setup_logger()
    logger.info(f"{'='*80}")
    logger.info(f"ECMWF 100米和60米风场数据处理器 - 批量处理 {len(elements)} 个要素")
    logger.info(f"要素列表: {', '.join(elements)}")
    logger.info(f"区域范围: {args.lon_west}E - {args.lon_east}E, {args.lat_south}N - {args.lat_north}N")
    logger.info(f"分辨率: {args.resolution}度")
    logger.info(f"60米风场计算方法: {args.calculation_method}")
    logger.info(f"幂律指数: {args.power_law_exponent}")
    logger.info(f"保存MICAPS4格式: {'是' if args.save_micaps4 else '否'}")
    if args.save_micaps4:
        if args.micaps4_output_dir:
            logger.info(f"MICAPS4输出目录: {args.micaps4_output_dir}")
        else:
            logger.info(f"MICAPS4输出目录: 使用NetCDF输出目录")
    logger.info(f"{'='*80}")
    
    # 创建处理器
    processor = EC_100m_60m_Processor(
        logger=logger, 
        config=config,
        save_micaps4=args.save_micaps4,
        micaps4_output_dir=args.micaps4_output_dir
    )
    
    total_stats = {
        'total_elements': 0,
        'success_elements': 0,
        'failed_elements': 0,
        'total_filesets': 0,
        'success_filesets': 0,
        'failed_filesets': 0,
        'total_micaps4_files': 0
    }
    
    try:
        # 循环处理每个要素
        for element in elements:
            total_stats['total_elements'] += 1
            logger.info(f"\n{'='*80}")
            logger.info(f"开始处理要素 [{total_stats['total_elements']}/{len(elements)}]: {element}")
            logger.info(f"{'='*80}")
            
            element_success = True
            
            try:
                # 处理方式1: 单个文件/文件集（仅支持单个要素）
                if args.input_dir is None and len(elements) == 1:
                    # 构建输入文件字典
                    input_files = {}
                    
                    if args.input_u_file and args.input_v_file:
                        input_files = {"u": args.input_u_file, "v": args.input_v_file}
                    else:
                        logger.error("风场要素需要U和V分量文件")
                        element_success = False
                        total_stats['failed_elements'] += 1
                        continue
                    
                    # 处理单个文件集
                    success, output_path, micaps4_files = processor.process_element(
                        element=element,
                        input_files=input_files,
                        output_dir=args.output_dir,
                        base_time=base_time,
                        skip_existing=args.skip_existing,
                        save_micaps4=args.save_micaps4,
                        micaps4_output_dir=args.micaps4_output_dir
                    )
                    
                    if success:
                        logger.info(f"要素 {element} 处理成功")
                        logger.info(f"NetCDF文件: {output_path}")
                        if micaps4_files:
                            micaps4_count = sum(len(files) for files in micaps4_files.values())
                            logger.info(f"MICAPS4文件: {micaps4_count} 个")
                            total_stats['total_micaps4_files'] += micaps4_count
                        total_stats['success_elements'] += 1
                    else:
                        logger.error(f"要素 {element} 处理失败")
                        element_success = False
                        total_stats['failed_elements'] += 1
                
                # 处理方式2: 目录批量处理
                else:
                    if args.input_dir is None:
                        logger.error(f"处理要素 {element} 需要输入目录")
                        element_success = False
                        total_stats['failed_elements'] += 1
                        continue
                    
                    logger.info(f"批量处理目录: {args.input_dir} - 要素: {element}")
                    
                    # 创建文件集
                    file_sets = create_file_sets_from_directory(args.input_dir, element)
                    
                    if not file_sets:
                        logger.error(f"在目录 {args.input_dir} 中未找到有效的文件集 - 要素: {element}")
                        element_success = False
                        total_stats['failed_elements'] += 1
                        continue
                    
                    logger.info(f"找到 {len(file_sets)} 个文件集")
                    total_stats['total_filesets'] += len(file_sets)
                    
                    # 批量处理
                    stats = processor.batch_process(
                        element=element,
                        file_sets=file_sets,
                        output_dir=args.output_dir,
                        skip_existing=args.skip_existing,
                        save_micaps4=args.save_micaps4,
                        micaps4_output_dir=args.micaps4_output_dir
                    )
                    
                    # 更新统计
                    total_stats['success_filesets'] += stats['success']
                    total_stats['failed_filesets'] += stats['failed']
                    total_stats['total_micaps4_files'] += stats['micaps4_files']
                    
                    # 记录处理结果
                    if stats['failed'] > 0:
                        logger.warning(f"要素 {element} 处理有 {stats['failed']} 个文件集失败")
                        element_success = False
                    else:
                        logger.info(f"要素 {element} 处理全部成功")
                    
                    if element_success:
                        total_stats['success_elements'] += 1
                    else:
                        total_stats['failed_elements'] += 1
            
            except Exception as e:
                logger.error(f"处理要素 {element} 时发生异常: {str(e)}", exc_info=True)
                total_stats['failed_elements'] += 1
                continue  # 继续处理下一个要素
        
        # 打印总体统计
        logger.info(f"\n{'='*80}")
        logger.info("所有要素处理完成！")
        logger.info(f"{'='*80}")
        logger.info(f"要素处理统计:")
        logger.info(f"  总要素数: {total_stats['total_elements']}")
        logger.info(f"  成功要素: {total_stats['success_elements']}")
        logger.info(f"  失败要素: {total_stats['failed_elements']}")
        
        if total_stats['total_filesets'] > 0:
            logger.info(f"文件集处理统计:")
            logger.info(f"  总文件集数: {total_stats['total_filesets']}")
            logger.info(f"  成功文件集: {total_stats['success_filesets']}")
            logger.info(f"  失败文件集: {total_stats['failed_filesets']}")
            success_rate = (total_stats['success_filesets'] / total_stats['total_filesets'] * 100) if total_stats['total_filesets'] > 0 else 0
            logger.info(f"  成功率: {success_rate:.1f}%")
        
        if args.save_micaps4:
            logger.info(f"  总MICAPS4文件数: {total_stats['total_micaps4_files']}")
            if args.micaps4_output_dir:
                logger.info(f"  MICAPS4文件目录: {args.micaps4_output_dir}")
                # 列出创建的要素文件夹
                if os.path.exists(args.micaps4_output_dir):
                    elements_found = [d for d in os.listdir(args.micaps4_output_dir) 
                                    if os.path.isdir(os.path.join(args.micaps4_output_dir, d))]
                    if elements_found:
                        logger.info(f"  创建的要素文件夹: {', '.join(elements_found)}")
        
        logger.info(f"{'='*80}")
        
        # 返回状态码：如果有任何要素失败，返回1
        return 0 if total_stats['failed_elements'] == 0 else 1
    
    except Exception as e:
        logger.error(f"处理过程中发生严重错误: {str(e)}", exc_info=True)
        return 1


# ================= 简化调用接口 =================
def process_multiple_elements(elements: List[str], input_dir: str, 
                             save_micaps4: bool = False, **kwargs) -> Dict[str, Any]:
    """
    简化接口：处理多个要素
    
    Parameters
    ----------
    elements : list of str
        要素名称列表
    input_dir : str
        输入目录
    save_micaps4 : bool
        是否保存MICAPS4格式
    **kwargs : dict
        其他参数
        
    Returns
    -------
    dict
        处理统计信息
    """
    processor = EC_100m_60m_Processor(save_micaps4=save_micaps4)
    
    total_stats = {
        'total_elements': len(elements),
        'success_elements': 0,
        'failed_elements': 0,
        'element_details': {}
    }
    
    for element in elements:
        try:
            file_sets = create_file_sets_from_directory(input_dir, element)
            
            if not file_sets:
                print(f"警告: 在目录 {input_dir} 中未找到有效的文件集 - 要素: {element}")
                total_stats['failed_elements'] += 1
                total_stats['element_details'][element] = {'status': 'failed', 'reason': 'no_files'}
                continue
            
            stats = processor.batch_process(
                element=element,
                file_sets=file_sets,
                save_micaps4=save_micaps4,
                **kwargs
            )
            
            if stats['failed'] == 0:
                total_stats['success_elements'] += 1
                total_stats['element_details'][element] = {'status': 'success', 'stats': stats}
            else:
                total_stats['failed_elements'] += 1
                total_stats['element_details'][element] = {'status': 'partial_failure', 'stats': stats}
                
        except Exception as e:
            total_stats['failed_elements'] += 1
            total_stats['element_details'][element] = {'status': 'failed', 'reason': str(e)}
    
    return total_stats

def process_single_element(element: str, input_files: Dict[str, str], 
                          save_micaps4: bool = False, **kwargs) -> Tuple[bool, str, Dict]:
    """
    简化接口：处理单个要素
    
    Parameters
    ----------
    element : str
        要素名称
    input_files : dict
        输入文件路径字典
    save_micaps4 : bool
        是否保存MICAPS4格式
    **kwargs : dict
        其他参数
        
    Returns
    -------
    tuple
        (是否成功, NetCDF输出文件路径, MICAPS4文件信息字典)
    """
    processor = EC_100m_60m_Processor(save_micaps4=save_micaps4)
    return processor.process_element(element, input_files, save_micaps4=save_micaps4, **kwargs)


def process_element_directory(element: str, input_dir: str, 
                             save_micaps4: bool = False, **kwargs) -> Dict[str, int]:
    """
    简化接口：处理目录中的所有要素文件
    
    Parameters
    ----------
    element : str
        要素名称
    input_dir : str
        输入目录
    save_micaps4 : bool
        是否保存MICAPS4格式
    **kwargs : dict
        其他参数
        
    Returns
    -------
    dict
        处理统计信息
    """
    processor = EC_100m_60m_Processor(save_micaps4=save_micaps4)
    file_sets = create_file_sets_from_directory(input_dir, element)
    
    return processor.batch_process(
        element=element,
        file_sets=file_sets,
        save_micaps4=save_micaps4,
        **kwargs
    )


if __name__ == "__main__":
    # 抑制警告
    warnings.filterwarnings("ignore")
    
    # 运行命令行接口
    sys.exit(main_cli())