#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECMWF 10米U/V风场数据业务化处理系统
任意时刻可调用的版本，支持NetCDF和MICAPS4格式输出

# 单个文件处理
python EC_processor.py \
  --input-u-file /path/to/ECMFC1D_10U_1_2026010100_GLB_1.grib1 \
  --input-v-file /path/to/ECMFC1D_10V_1_2026010100_GLB_1.grib1 \
  --output-dir /path/to/output

python aa.py   --input-u-file /home/youqi/FWZX_forecast_DATA/data_demo/ECMFC1D_10U_1_2026012012_GLB_1.grib1 \
  --input-v-file /home/youqi/FWZX_forecast_DATA/data_demo/ECMFC1D_10V_1_2026012012_GLB_1.grib1  \
   --save-micaps4  \
    --micaps4-output-dir /home/youqi/FWZX_forecast_DATA/tonst


# 保存MICAPS4格式
python EC_processor.py \
  --input-u-file /path/to/U.grib1 \
  --input-v-file /path/to/V.grib1 \
  --save-micaps4

# 在Python代码中直接调用
from EC_processor import ECProcessor
processor = ECProcessor(save_micaps4=True)
success, nc_file, micaps4_files = processor.process_file(
    u_file="/path/to/U.grib1",
    v_file="/path/to/V.grib1"
)
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
from pathlib import Path
from typing import Tuple, List, Dict, Optional

# ================= 配置类 =================
class Config:
    """集中管理所有配置参数"""
    # 区域范围
    REGION = {
        "lon_w": 110.0,
        "lon_e": 121.0,
        "lat_s": 35.0,
        "lat_n": 44.0
    }
    
    # 输出分辨率
    RESOLUTION = 0.01
    
    # 时区偏移（北京时）
    TIMEZONE_SHIFT = timedelta(hours=8)
    
    # 默认输出文件名格式
    OUTPUT_FILENAME_FORMAT = "wspd_wdir_10m_0p01_1h_BJT_{time_str}.nc"
    
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




# ================= MICAPS4标准要素名称 =================
class MICAPS4Elements:
    """MICAPS4标准要素名称定义"""
    
    # 根据文档第7-8页，风向风速类201-400
    WIND_DIRECTION = "WINDD"  # 风向，对应ID 201
    WIND_SPEED = "WINDS"      # 风速，对应ID 203
    # 也可以使用更具体的名称
    WIND_DIR_10M = "WINDD_10M"  # 10米风向
    WIND_SPD_10M = "WINDS_10M"  # 10米风速
    
    # 其他常用要素示例
    TEMPERATURE = "TEMPERATURE"  # 温度
    PRESSURE = "PRESSURE"        # 气压
    HUMIDITY = "HUMIDITY"        # 湿度
    
    @staticmethod
    def get_element_id(element_name: str) -> int:
        """获取要素对应的标准ID"""
        element_map = {
            "WINDD": 201,      # 风向
            "WINDS": 203,      # 风速
            "WINDD_10M": 205,  # 10米风向（使用接近的ID）
            "WINDS_10M": 207,  # 10米风速（使用接近的ID）
        }
        return element_map.get(element_name.upper(), 0)


# ================= MICAPS4格式保存类（二进制格式，完全符合MICAPS4标准） =================
class MICAPS4Writer:
    """MICAPS第4类数据格式（网格数据）写入器 - 二进制格式，完全符合MICAPS4标准"""
    
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
            要素名称，使用标准名称
        model_name : str
            模式名称，如 "ECMWF", "GRAPES"
        timezone_shift : timedelta, optional
            时区偏移，如为None则使用北京时
            
        Returns
        -------
        str
            MICAPS4格式文件名（传统格式：YYYYMMDDHH.TTT）
        """
        if timezone_shift is None:
            timezone_shift = Config.TIMEZONE_SHIFT
        
        # 转换为本地时间
        local_time = base_time + timezone_shift
        
        # MICAPS传统命名：YYYYMMDDHH.TTT
        # 例如：2024010100.012 表示2024年1月1日00时起报的12小时预报
        return f"{local_time.strftime('%Y%m%d%H')}.{forecast_hour:03d}"
    
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
        
        注意：这是矢量数据格式，不是两个单独的标量文件！
        
        Parameters
        ----------
        wspd : np.ndarray
            风速数据 (m/s)
        wdir : np.ndarray
            风向数据 (度)，气象标准：0°=北风，90°=东风，180°=南风，270°=西风
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
                # ============= 1. discriminator: 始终为小写的mdfs，字符型4字节 =============
                f.write(b'mdfs')
                
                # ============= 2. type: short型2字节，11为模式矢量数据 =============
                data_type = 11  # 矢量网格数据
                f.write(np.int16(data_type).tobytes())
                
                # ============= 3. modelName: 字节型20字节，全大写字母，不足补0 =============
                model_name_bytes = model_name.upper().encode('ascii', 'ignore')[:20]
                if len(model_name_bytes) < 20:
                    model_name_bytes += b'\x00' * (20 - len(model_name_bytes))
                f.write(model_name_bytes)
                
                # ============= 4. element: 字节型50字节，必须使用标准名称 =============
                element_std = "WIND_10M"
                element_bytes = element_std.encode('ascii', 'ignore')[:50]
                if len(element_bytes) < 50:
                    element_bytes += b'\x00' * (50 - len(element_bytes))
                f.write(element_bytes)
                
                # ============= 5. description: 字节型30字节，用于表示附加描述 =============
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
                
                # ============= 6. level: 层次，float型4字节 =============
                f.write(np.float32(level).tobytes())
                
                # ============= 7-10. 起报日期和时间 =============
                f.write(np.int32(local_time.year).tobytes())
                f.write(np.int32(local_time.month).tobytes())
                f.write(np.int32(local_time.day).tobytes())
                f.write(np.int32(local_time.hour).tobytes())
                
                # ============= 11. timezone: 时区，int型4字节 =============
                timezone = 8  # 北京时区
                f.write(np.int32(timezone).tobytes())
                
                # ============= 12. period: 预报时效，单位小时，int型4字节 =============
                f.write(np.int32(forecast_hour).tobytes())
                
                # ============= 13-15. 经度范围信息 =============
                f.write(np.float32(start_lon).tobytes())
                f.write(np.float32(end_lon).tobytes())
                f.write(np.float32(lon_res).tobytes())
                
                # ============= 16. longitudeGridNumber: 纬向经线格点数 =============
                f.write(np.int32(n_lon).tobytes())
                
                # ============= 17-19. 纬度范围信息 =============
                f.write(np.float32(start_lat).tobytes())
                f.write(np.float32(end_lat).tobytes())
                f.write(np.float32(lat_res).tobytes())
                
                # ============= 20. latitudeGridNumber: 经向纬线格点数 =============
                f.write(np.int32(n_lat).tobytes())
                
                # ============= 21-23. 等值线相关信息（矢量数据应忽略） =============
                f.write(np.float32(0.0).tobytes())  # isolineStartValue
                f.write(np.float32(0.0).tobytes())  # isolineEndValue
                f.write(np.float32(0.0).tobytes())  # isolineSpace
                
                # ============= 24. Extent: 扩展段，100字节，设置为全0 =============
                f.write(b'\x00' * 100)
                
                # ============= 数据区（矢量数据特殊格式） =============
                # 处理NaN值
                wspd_clean = wspd.copy()
                wdir_clean = wdir.copy()
                
                nan_mask_wspd = np.isnan(wspd_clean)
                nan_mask_wdir = np.isnan(wdir_clean)
                
                if np.any(nan_mask_wspd):
                    wspd_clean[nan_mask_wspd] = 9999.0
                if np.any(nan_mask_wdir):
                    wdir_clean[nan_mask_wdir] = 9999.0
                
                # ============= 关键修改：风向转换 =============
                # 气象风向：0°=北风，90°=东风，180°=南风，270°=西风
                # MICAPS矢量角度：0°=西风，90°=南风，180°=东风，270°=北风
                # 转换公式：MICAPS角度 = (270 - 气象风向) mod 360
                micaps_wdir = (270.0 - wdir_clean) % 360.0
                
                # 确保风向在0-360度范围内
                micaps_wdir = micaps_wdir % 360.0
                
                # 展平数据（按MICAPS4要求的顺序：先纬向后经向）
                wspd_flat = wspd_clean.ravel(order='C').astype(np.float32)
                wdir_flat = micaps_wdir.ravel(order='C').astype(np.float32)
                
                # 写入风速（模）
                f.write(wspd_flat.tobytes())
                # 写入风向（MICAPS角度）
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
            
            # 添加风向转换说明
            print(f"  风向转换: 气象风向 → MICAPS矢量角度")
            print(f"    0°(北风) → 270°")
            print(f"    90°(东风) → 180°") 
            print(f"    180°(南风) → 90°")
            print(f"    270°(西风) → 0°")
            
            return True
            
        except Exception as e:
            print(f"✗ 写入MICAPS4矢量文件失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    @staticmethod
    def write_wind_vector_file(wspd: np.ndarray, wdir: np.ndarray, 
                              lats: np.ndarray, lons: np.ndarray,
                              base_time: datetime, forecast_hour: int,
                              output_dir: str, model_name: str = "ECMWF",
                              timezone_shift: timedelta = None) -> Dict[str, str]:
        """
        写入风速和风向的MICAPS4矢量文件（单个文件，包含模和角度）
        
        这是MICAPS4正确的做法：风速和风向在一个文件中（type=11）
        
        Parameters
        ----------
        wspd : np.ndarray
            风速数据 (m/s)
        wdir : np.ndarray
            风向数据 (度)
        lats : np.ndarray
            纬度数组
        lons : np.ndarray
            经度数组
        base_time : datetime
            预报起始时间（UTC）
        forecast_hour : int
            预报时效
        output_dir : str
            输出目录
        model_name : str
            模式名称
        timezone_shift : timedelta
            时区偏移
            
        Returns
        -------
        dict
            生成的文件路径字典
        """
        if timezone_shift is None:
            timezone_shift = Config.TIMEZONE_SHIFT
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建文件名（使用传统的MICAPS命名）
        filename = MICAPS4Writer.create_micaps4_filename(base_time, forecast_hour, 'WIND', model_name, timezone_shift)
        output_path = os.path.join(output_dir, filename)
        
        result = {}
        
        success = MICAPS4Writer.write_micaps4_vector_file(
            wspd=wspd, wdir=wdir, lats=lats, lons=lons,
            base_time=base_time, forecast_hour=forecast_hour,
            output_path=output_path, model_name=model_name,
            description="10m wind speed and direction",
            timezone_shift=timezone_shift
        )
        
        if success:
            result['wind_vector'] = output_path
            
        return result


# ================= 日志配置 =================
def setup_logger(log_level=logging.INFO, name='ECMWF_Processor'):
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
class ECProcessor:
    """ECMWF风场数据处理器 - 业务化版本（支持NetCDF和MICAPS4输出）"""
    
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
        
    def process_file(self, u_file: str, v_file: str, 
                    output_dir: str = None,
                    base_time: datetime = None,
                    output_filename: str = None,
                    skip_existing: bool = True,
                    save_micaps4: bool = None,
                    micaps4_output_dir: str = None) -> Tuple[bool, str, Dict]:
        """
        处理单个ECMWF风场数据文件对
        
        Parameters
        ----------
        u_file : str
            U分量文件路径
        v_file : str
            V分量文件路径
        output_dir : str, optional
            输出目录，如为None则输出到输入文件同目录
        base_time : datetime, optional
            基准时间（UTC），如为None则从文件名解析
        output_filename : str, optional
            输出文件名，如为None则自动生成
        skip_existing : bool, optional
            是否跳过已存在的输出文件，默认为True
        save_micaps4 : bool, optional
            是否保存MICAPS4格式，如为None则使用类设置
        micaps4_output_dir : str, optional
            MICAPS4输出目录，如为None则使用类设置
            
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
            # 1. 检查输入文件
            u_basename = os.path.basename(u_file)
            self.logger.info(f"开始处理: {u_basename}")
            
            if not os.path.exists(u_file):
                self.logger.error(f"U文件不存在: {u_file}")
                return False, None, {}
                
            if not os.path.exists(v_file):
                self.logger.error(f"V文件不存在: {v_file}")
                return False, None, {}
            
            # 2. 确定基准时间
            if base_time is None:
                base_time = Config.parse_time_from_filename(u_file)
                if base_time is None:
                    base_time = Config.parse_time_from_filename(v_file)
            
            if base_time is None:
                self.logger.error("无法从文件名解析时间，请提供base_time参数")
                return False, None, {}
            
            # 3. 确定输出路径
            if output_dir is None:
                output_dir = os.path.dirname(u_file)
            
            if output_filename is None:
                bjt_time_str = Config.utc_to_bjt_str(base_time)
                output_filename = Config.OUTPUT_FILENAME_FORMAT.format(time_str=bjt_time_str)
            
            output_path = os.path.join(output_dir, output_filename)
            
            # 4. 检查输出是否已存在
            if skip_existing and os.path.exists(output_path):
                self.logger.info(f"输出文件已存在: {output_path}")
                return True, output_path, {}
            
            # 5. 执行处理（包含MICAPS4输出）
            success, wspd_1h, wdir_1h, times_bjt = self._process_core_with_micaps4(
                u_file, v_file, base_time, output_path, save_micaps4, micaps4_output_dir
            )
            
            if success and save_micaps4 and wspd_1h is not None and wdir_1h is not None:
                # 保存MICAPS4格式文件（使用矢量格式）
                micaps4_files = self._save_micaps4_files(
                    wspd_1h, wdir_1h, base_time, micaps4_output_dir
                )
            
            if success:
                total_time = time.time() - total_start
                self.logger.info(f"处理完成，总耗时: {total_time:.1f}秒")
                self.logger.info(f"NetCDF输出文件: {output_path}")
                
                if micaps4_files:
                    self.logger.info(f"MICAPS4矢量文件: {len(micaps4_files.get('wind_vector', []))} 个")
            
            return success, output_path, micaps4_files
            
        except Exception as e:
            self.logger.error(f"处理失败: {str(e)}", exc_info=True)
            return False, output_path, micaps4_files
    
    def _process_core_with_micaps4(self, u_file: str, v_file: str, 
                                  base_time_utc: datetime, output_path: str,
                                  save_micaps4: bool, micaps4_output_dir: str) -> Tuple[bool, np.ndarray, np.ndarray, list]:
        """核心处理逻辑，返回风速风向数据用于MICAPS4输出"""
        try:
            self.logger.info("开始处理数据...")
            
            # 1. 读取数据
            start_time = time.time()
            with pygrib.open(u_file) as grbs_u, pygrib.open(v_file) as grbs_v:
                # 获取所有时次
                steps_u = sorted({msg.step for msg in grbs_u})
                steps_v = sorted({msg.step for msg in grbs_v})
                common_steps = sorted(set(steps_u) & set(steps_v))
                
                if not common_steps:
                    self.logger.error("没有共同的预报时次")
                    return False, None, None, None
                
                # 读取第一个消息获取网格信息
                msg_u = grbs_u.select(step=common_steps[0])[0]
                data, lats, lons = msg_u.data(
                    lat1=Config.REGION["lat_s"],
                    lat2=Config.REGION["lat_n"],
                    lon1=Config.REGION["lon_w"],
                    lon2=Config.REGION["lon_e"]
                )
                
                # 获取源网格
                lat_src = lats[:, 0]  # 纬度数组
                lon_src = lons[0, :]  # 经度数组
                
                # 检查源网格纬度顺序
                if lat_src[-1] < lat_src[0]:  # 如果源纬度是降序
                    self.logger.info(f"源纬度是降序，需要反转")
                    lat_src = lat_src[::-1]  # 反转成升序
                
                self.logger.info(f"目标纬度: {self.lat_dst[0]:.2f} 到 {self.lat_dst[-1]:.2f}")
                self.logger.info(f"源纬度: {lat_src[0]:.2f} 到 {lat_src[-1]:.2f}")
                
                # 重新读取所有数据
                u_cube = []
                for step in common_steps:
                    msg = grbs_u.select(step=step)[0]
                    data, lats, lons = msg.data(
                        lat1=Config.REGION["lat_s"],
                        lat2=Config.REGION["lat_n"],
                        lon1=Config.REGION["lon_w"],
                        lon2=Config.REGION["lon_e"]
                    )
                    
                    # 确保数据与源网格纬度方向一致（升序）
                    if lat_src[-1] > lat_src[0]:  # 如果源纬度是升序
                        if lats[-1, 0] < lats[0, 0]:  # 如果数据纬度是降序
                            data = data[::-1, :]
                    else:  # 如果源纬度是降序
                        if lats[-1, 0] > lats[0, 0]:  # 如果数据纬度是升序
                            data = data[::-1, :]
                    
                    u_cube.append(data)
                
                u_cube = np.array(u_cube, dtype=np.float32)
                
                # 读取V数据
                v_cube = []
                for step in common_steps:
                    msg = grbs_v.select(step=step)[0]
                    data, lats, lons = msg.data(
                        lat1=Config.REGION["lat_s"],
                        lat2=Config.REGION["lat_n"],
                        lon1=Config.REGION["lon_w"],
                        lon2=Config.REGION["lon_e"]
                    )
                    
                    # 确保数据与源网格纬度方向一致
                    if lat_src[-1] > lat_src[0]:  # 如果源纬度是升序
                        if lats[-1, 0] < lats[0, 0]:  # 如果数据纬度是降序
                            data = data[::-1, :]
                    else:  # 如果源纬度是降序
                        if lats[-1, 0] > lats[0, 0]:  # 如果数据纬度是升序
                            data = data[::-1, :]
                    
                    v_cube.append(data)
                
                v_cube = np.array(v_cube, dtype=np.float32)
            
            read_time = time.time() - start_time
            self.logger.info(f"数据读取完成: {read_time:.1f}秒, 时次: {len(common_steps)}")
            
            # 2. 空间插值
            start_time = time.time()
            
            # 准备目标网格点
            lon2d, lat2d = np.meshgrid(self.lon_dst, self.lat_dst)
            points = np.column_stack([lat2d.ravel(), lon2d.ravel()])
            
            # 创建插值器（要求lat_src和lon_src都是升序）
            interp_func = RegularGridInterpolator(
                (lat_src, lon_src),
                u_cube[0],
                bounds_error=False,
                fill_value=np.nan
            )
            
            # 批量插值
            u_interp = np.empty((len(common_steps), self.n_lat_dst, self.n_lon_dst), dtype=np.float32)
            v_interp = np.empty_like(u_interp)
            
            for i in range(len(common_steps)):
                interp_func.values = u_cube[i]
                u_interp[i] = interp_func(points).reshape(self.n_lat_dst, self.n_lon_dst)
                interp_func.values = v_cube[i]
                v_interp[i] = interp_func(points).reshape(self.n_lat_dst, self.n_lon_dst)
            
            interp_time = time.time() - start_time
            self.logger.info(f"空间插值完成: {interp_time:.1f}秒")
            
            # 3. 计算风速风向
            start_time = time.time()
            wspd = np.sqrt(u_interp**2 + v_interp**2)
            wdir = (np.degrees(np.arctan2(u_interp, v_interp)) + 180) % 360
            calc_time = time.time() - start_time
            self.logger.info(f"风速风向计算完成: {calc_time:.1f}秒")
            
            # 4. 时间插值
            start_time = time.time()
            hours = np.arange(len(common_steps)) * 3  # 假设3小时间隔
            hours_new = np.arange(0, hours[-1] + 1, 1.0)
            
            f_wspd = interp1d(hours, wspd, axis=0, kind='linear')
            f_wdir = interp1d(hours, wdir, axis=0, kind='linear')
            
            wspd_1h = f_wspd(hours_new)
            wdir_1h = f_wdir(hours_new)
            
            # 生成北京时时间序列
            times_bjt = [base_time_utc + timedelta(hours=float(h)) + Config.TIMEZONE_SHIFT 
                        for h in hours_new]
            time_interp_time = time.time() - start_time
            self.logger.info(f"时间插值完成: {time_interp_time:.1f}秒, {len(common_steps)}->{len(times_bjt)}时次")
            
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
                wspd_var = nc.createVariable('wspd', 'f4', ('time', 'lat', 'lon'),
                                             zlib=True, complevel=1)
                wdir_var = nc.createVariable('wdir', 'f4', ('time', 'lat', 'lon'),
                                             zlib=True, complevel=1)
                
                # 设置数据（纬度已经是升序）
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
                wspd_var.long_name = '10 meter wind speed'
                
                wdir_var[:] = wdir_1h
                wdir_var.units = 'degree'
                wdir_var.long_name = '10 meter wind direction'
                wdir_var.comment = '0° indicates wind from north, 90° from east'
                
                # 添加全局属性
                nc.title = 'ECMWF 10m Wind Speed and Direction (Beijing Time)'
                nc.source = 'ECMWF Forecast'
                nc.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                nc.forecast_start_time_utc = base_time_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
                nc.forecast_start_time_bjt = Config.utc_to_bjt_str(base_time_utc)
                nc.latitude_order = 'south to north (ascending)'
            
            write_time = time.time() - start_time
            file_size_mb = os.path.getsize(output_path) / 1024 / 1024
            self.logger.info(f"NetCDF文件写入完成: {write_time:.1f}秒, 大小: {file_size_mb:.1f} MB")
            
            return True, wspd_1h, wdir_1h, times_bjt
            
        except Exception as e:
            self.logger.error(f"处理失败: {str(e)}", exc_info=True)
            return False, None, None, None
    
    def _save_micaps4_files(self, wspd_1h: np.ndarray, wdir_1h: np.ndarray,
                           base_time_utc: datetime, output_dir: str) -> Dict[str, List[str]]:
        """
        保存MICAPS4格式文件（使用矢量格式：风速和风向在一个文件中）
        
        Parameters
        ----------
        wspd_1h : np.ndarray
            1小时间隔的风速数据
        wdir_1h : np.ndarray
            1小时间隔的风向数据
        base_time_utc : datetime
            UTC基准时间
        output_dir : str
            输出目录
            
        Returns
        -------
        dict
            文件路径字典 {'wind_vector': [file1, file2, ...]}
        """
        if not output_dir:
            self.logger.warning("未指定MICAPS4输出目录，跳过保存")
            return {'wind_vector': []}
        
        self.logger.info("开始保存MICAPS4矢量格式文件...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        micaps4_files = {'wind_vector': []}
        
        start_time = time.time()
        
        # 保存每个时次的数据（使用矢量格式：风速和风向在一个文件中）
        for i in range(wspd_1h.shape[0]):
            forecast_hour = i  # 预报时效（小时）
            
            # 创建文件名
            filename = MICAPS4Writer.create_micaps4_filename(
                base_time=base_time_utc,
                forecast_hour=forecast_hour,
                element="WIND",
                model_name="ECMWF"
            )
            output_path = os.path.join(output_dir, filename)
            
            # 检查数据是否有有效值
            if np.all(np.isnan(wspd_1h[i])) or np.all(np.isnan(wdir_1h[i])):
                self.logger.warning(f"风速或风向数据全为NaN，跳过时次 {forecast_hour}")
                continue
                
            success = MICAPS4Writer.write_micaps4_vector_file(
                wspd=wspd_1h[i], wdir=wdir_1h[i],
                lats=self.lat_dst, lons=self.lon_dst,
                base_time=base_time_utc, forecast_hour=forecast_hour,
                output_path=output_path, model_name="ECMWF",
                description="10m wind speed and direction"
            )
            
            if success:
                micaps4_files['wind_vector'].append(output_path)
            else:
                self.logger.warning(f"MICAPS4矢量文件保存失败: 时次 {forecast_hour}")
            
            # 进度显示
            if (i + 1) % 10 == 0 or (i + 1) == wspd_1h.shape[0]:
                saved_count = len(micaps4_files['wind_vector'])
                self.logger.info(f"  MICAPS4进度: {i+1}/{wspd_1h.shape[0]} 时次 ({saved_count}个矢量文件)")
        
        total_time = time.time() - start_time
        total_files = len(micaps4_files['wind_vector'])
        self.logger.info(f"MICAPS4矢量文件保存完成: 共{total_files}个文件, 耗时: {total_time:.1f}秒")
        
        return micaps4_files
    
    def batch_process(self, file_pairs: List[Tuple[str, str]], 
                     output_dir: str = None,
                     base_times: List[datetime] = None,
                     skip_existing: bool = True,
                     save_micaps4: bool = None,
                     micaps4_output_dir: str = None) -> Dict[str, int]:
        """
        批量处理多个文件对
        
        Parameters
        ----------
        file_pairs : list of tuple
            文件对列表，每个元素为(u_file, v_file)
        output_dir : str, optional
            输出目录，如为None则输出到输入文件同目录
        base_times : list of datetime, optional
            基准时间列表，如为None则从文件名解析
        skip_existing : bool, optional
            是否跳过已存在的输出文件，默认为True
        save_micaps4 : bool, optional
            是否保存MICAPS4格式，如为None则使用类设置
        micaps4_output_dir : str, optional
            MICAPS4输出目录
            
        Returns
        -------
        dict
            处理统计信息
        """
        # 确定是否保存MICAPS4
        if save_micaps4 is None:
            save_micaps4 = self.save_micaps4
        
        stats = {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0, 'micaps4_files': 0}
        
        total_start = time.time()
        self.logger.info(f"开始批量处理 {len(file_pairs)} 个文件对")
        self.logger.info(f"保存MICAPS4格式: {'是' if save_micaps4 else '否'}")
        
        for i, (u_file, v_file) in enumerate(file_pairs):
            stats['total'] += 1
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"处理文件对 {i+1}/{len(file_pairs)}")
            self.logger.info(f"U文件: {os.path.basename(u_file)}")
            self.logger.info(f"V文件: {os.path.basename(v_file)}")
            
            # 确定基准时间
            base_time = None
            if base_times and i < len(base_times):
                base_time = base_times[i]
            
            # 处理文件
            success, output_path, micaps4_files = self.process_file(
                u_file=u_file,
                v_file=v_file,
                output_dir=output_dir,
                base_time=base_time,
                skip_existing=skip_existing,
                save_micaps4=save_micaps4,
                micaps4_output_dir=micaps4_output_dir
            )
            
            if success:
                stats['success'] += 1
                if micaps4_files:
                    stats['micaps4_files'] += len(micaps4_files.get('wind_vector', []))
            else:
                stats['failed'] += 1
            
            # 短暂休息，避免资源竞争
            time.sleep(0.1)
        
        # 打印统计
        total_time = time.time() - total_start
        self.logger.info(f"\n{'='*80}")
        self.logger.info("批量处理完成！")
        self.logger.info(f"总文件对: {stats['total']}")
        self.logger.info(f"成功: {stats['success']}")
        self.logger.info(f"失败: {stats['failed']}")
        self.logger.info(f"跳过: {stats['skipped']}")
        if save_micaps4:
            self.logger.info(f"MICAPS4矢量文件: {stats['micaps4_files']} 个")
        self.logger.info(f"总耗时: {total_time:.1f}秒")
        if stats['total'] > 0:
            self.logger.info(f"平均时间: {total_time/stats['total']:.1f}秒/文件对")
        self.logger.info(f"{'='*80}")
        
        return stats


# ================= 辅助函数 =================
def create_file_pairs_from_directory(input_dir: str, pattern: str = "ECMFC1D_*_*.grib1") -> List[Tuple[str, str]]:
    """
    从目录创建文件对
    
    Parameters
    ----------
    input_dir : str
        输入目录
    pattern : str
        文件匹配模式
        
    Returns
    -------
    list
        文件对列表 [(u_file1, v_file1), (u_file2, v_file2), ...]
    """
    from glob import glob
    
    # 获取所有U分量文件
    u_files = sorted(glob(os.path.join(input_dir, "ECMFC1D_10U_*.grib1")))
    file_pairs = []
    
    for u_file in u_files:
        # 从U文件名生成V文件名
        v_file = u_file.replace("_10U_", "_10V_")
        if os.path.exists(v_file):
            file_pairs.append((u_file, v_file))
        else:
            print(f"警告: 找不到对应的V文件: {v_file}")
    
    return file_pairs


# ================= 命令行接口 =================
def main_cli():
    """命令行接口主函数 - 支持MICAPS4输出"""
    parser = argparse.ArgumentParser(description='ECMWF风场数据处理器 - 支持NetCDF和MICAPS4输出')
    
    # 输入方式1: 单个文件对
    parser.add_argument('--input-u-file', type=str, default=None,
                       help='U分量文件路径')
    
    parser.add_argument('--input-v-file', type=str, default=None,
                       help='V分量文件路径')
    
    # 输入方式2: 目录批量处理
    parser.add_argument('--input-dir', type=str, default=None,
                       help='输入目录（包含U/V文件）')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default=None,
                       help='NetCDF输出目录，如为None则输出到输入文件同目录')
    
    parser.add_argument('--base-time', type=str, default=None,
                       help='基准时间（UTC），格式: YYYYMMDDHH，如为None则从文件名解析')
    
    # MICAPS4相关参数
    parser.add_argument('--save-micaps4', action='store_true', default=False,
                       help='保存MICAPS4格式文件')
    
    parser.add_argument('--micaps4-output-dir', type=str, default=None,
                       help='MICAPS4输出目录，如为None则与NetCDF输出同目录')
    
    parser.add_argument('--micaps4-prefix', type=str, default="ECMWF",
                       help='MICAPS4文件名前缀')
    
    # 处理选项
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='跳过已存在的输出文件（默认启用）')
    
    parser.add_argument('--no-skip-existing', action='store_false', dest='skip_existing',
                       help='不跳过已存在的输出文件')


    
    # 配置覆盖
    parser.add_argument('--lon-west', type=float, default=110.0,
                       help='区域西边界经度')
    
    parser.add_argument('--lon-east', type=float, default=121.0,
                       help='区域东边界经度')
    
    parser.add_argument('--lat-south', type=float, default=35.0,
                       help='区域南边界纬度')
    
    parser.add_argument('--lat-north', type=float, default=44.0,
                       help='区域北边界纬度')
    
    parser.add_argument('--resolution', type=float, default=0.01,
                       help='输出分辨率（度）')
    
    args = parser.parse_args()
    
    # 检查输入参数
    if args.input_u_file is None and args.input_dir is None:
        print("错误: 必须提供--input-u-file或--input-dir参数")
        parser.print_help()
        return 1
    
    if args.input_u_file is not None and args.input_v_file is None:
        print("错误: 提供了--input-u-file但未提供--input-v-file")
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
        "RESOLUTION": args.resolution
    }
    
    # 设置日志
    logger = setup_logger()
    logger.info(f"{'='*80}")
    logger.info("ECMWF风场数据处理器 - 支持NetCDF和MICAPS4输出")
    logger.info(f"区域范围: {args.lon_west}E - {args.lon_east}E, {args.lat_south}N - {args.lat_north}N")
    logger.info(f"分辨率: {args.resolution}度")
    logger.info(f"保存MICAPS4格式: {'是' if args.save_micaps4 else '否'}")
    if args.save_micaps4:
        logger.info(f"MICAPS4输出目录: {args.micaps4_output_dir or args.output_dir or '自动'}")
    logger.info(f"{'='*80}")
    
    # 创建处理器
    processor = ECProcessor(
        logger=logger, 
        config=config,
        save_micaps4=args.save_micaps4,
        micaps4_output_dir=args.micaps4_output_dir
    )
    
    try:
        # 处理方式1: 单个文件对
        if args.input_u_file:
            logger.info(f"处理单个文件对")
            logger.info(f"U文件: {args.input_u_file}")
            logger.info(f"V文件: {args.input_v_file}")
            
            success, output_path, micaps4_files = processor.process_file(
                u_file=args.input_u_file,
                v_file=args.input_v_file,
                output_dir=args.output_dir,
                base_time=base_time,
                skip_existing=args.skip_existing,
                save_micaps4=args.save_micaps4,
                micaps4_output_dir=args.micaps4_output_dir
            )
            
            if success:
                logger.info(f"处理成功")
                logger.info(f"NetCDF文件: {output_path}")
                if micaps4_files:
                    logger.info(f"MICAPS4矢量文件: {len(micaps4_files.get('wind_vector', []))} 个")
                return 0
            else:
                logger.error("处理失败")
                return 1
        
        # 处理方式2: 目录批量处理
        elif args.input_dir:
            logger.info(f"批量处理目录: {args.input_dir}")
            
            # 创建文件对
            file_pairs = create_file_pairs_from_directory(args.input_dir)
            
            if not file_pairs:
                logger.error(f"在目录 {args.input_dir} 中未找到有效的文件对")
                return 1
            
            logger.info(f"找到 {len(file_pairs)} 个文件对")
            
            # 批量处理
            stats = processor.batch_process(
                file_pairs=file_pairs,
                output_dir=args.output_dir,
                skip_existing=args.skip_existing,
                save_micaps4=args.save_micaps4,
                micaps4_output_dir=args.micaps4_output_dir
            )
            
            return 0 if stats['failed'] == 0 else 1
    
    except Exception as e:
        logger.error(f"处理失败: {str(e)}", exc_info=True)
        return 1


# ================= 简化调用接口 =================
def process_single_file(u_file: str, v_file: str, save_micaps4: bool = False, **kwargs) -> Tuple[bool, str, Dict]:
    """
    简化接口：处理单个文件对（支持MICAPS4）
    
    Parameters
    ----------
    u_file : str
        U分量文件路径
    v_file : str
        V分量文件路径
    save_micaps4 : bool
        是否保存MICAPS4格式
    **kwargs : dict
        其他参数，传递给ECProcessor.process_file
        
    Returns
    -------
    tuple
        (是否成功, NetCDF输出文件路径, MICAPS4文件信息字典)
    """
    processor = ECProcessor(save_micaps4=save_micaps4)
    return processor.process_file(u_file, v_file, save_micaps4=save_micaps4, **kwargs)


def process_directory(input_dir: str, save_micaps4: bool = False, **kwargs) -> Dict[str, int]:
    """
    简化接口：处理目录中的所有文件对（支持MICAPS4）
    
    Parameters
    ----------
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
    processor = ECProcessor(save_micaps4=save_micaps4)
    file_pairs = create_file_pairs_from_directory(input_dir)
    
    return processor.batch_process(
        file_pairs=file_pairs,
        save_micaps4=save_micaps4,
        **kwargs
    )


if __name__ == "__main__":
    # 抑制警告
    warnings.filterwarnings("ignore")
    
    # 运行命令行接口
    sys.exit(main_cli())