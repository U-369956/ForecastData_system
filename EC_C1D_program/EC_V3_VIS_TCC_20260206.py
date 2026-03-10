#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECMWF VIS和TCC数据处理模块 - 基于简化版思路
支持MICAPS4格式输出
"""

import os
import sys
import numpy as np
from datetime import datetime, timedelta
import pygrib
from scipy.interpolate import RegularGridInterpolator, interp1d
from netCDF4 import Dataset, date2num
import time
from typing import Tuple, List, Dict, Optional
import warnings
import argparse

# 抑制警告
warnings.filterwarnings("ignore")

# ================= 配置类 =================
class Config:
    """集中管理所有配置参数"""
    # 区域范围
    REGION = {
        "lon_w": 110.0,
        "lon_e": 125.0,
        "lat_s": 34.0,
        "lat_n": 44.0
    }
    
    # 输出分辨率
    RESOLUTION = 0.01
    
    # 时区偏移（北京时）
    TIMEZONE_SHIFT = timedelta(hours=8)
    
    # 默认输出文件名格式
    OUTPUT_FILENAME_FORMAT = "{element}_0p01_1h_BJT_{time_str}.nc"
    
    # 要素配置 - 扩展VIS和TCC
    ELEMENTS = {
        # 原有要素...
        "WIND": {
            "description": "10m wind speed and direction",
            "grib_codes": {"u": "10U", "v": "10V"},
            "requires_uv": True,
            "output_vars": ["wspd", "wdir"],
            "units": {"wspd": "m s-1", "wdir": "degree"},
            "micaps_type": "vector",  # MICAPS数据类型：vector或scalar
            "micaps_element": "10WIND"  # MICAPS要素名
        },
        "GUST": {
            "description": "10m wind gust",
            "grib_codes": {"value": "10FG3"},
            "requires_uv": False,
            "output_vars": ["gust"],
            "units": {"gust": "m s-1"},
            "micaps_type": "scalar",
            "micaps_element": "GUST"
        },
        "TEM": {
            "description": "2m temperature",
            "grib_codes": {"value": "TEM"},
            "requires_uv": False,
            "output_vars": ["temp"],
            "units": {"temp": "°C"},
            "conversion": "K_to_C",
            "micaps_type": "scalar",
            "micaps_element": "TEM"
        },
        "PRS": {
            "description": "Surface pressure",
            "grib_codes": {"value": "PRS"},
            "requires_uv": False,
            "output_vars": ["prs"],
            "units": {"prs": "Pa"},
            "micaps_type": "scalar",
            "micaps_element": "PRS"
        },
        "DPT": {
            "description": "2m dew point temperature",
            "grib_codes": {"value": "DPT"},
            "requires_uv": False,
            "output_vars": ["dpt"],
            "units": {"dpt": "°C"},
            "conversion": "K_to_C",
            "micaps_type": "scalar",
            "micaps_element": "DPT"
        },
        "RH": {
            "description": "2m relative humidity",
            "grib_codes": {"temp": "TEM", "prs": "PRS", "dpt": "DPT"},
            "requires_uv": False,
            "requires_calc": True,
            "output_vars": ["rh"],
            "units": {"rh": "%"},
            "micaps_type": "scalar",
            "micaps_element": "RH"
        },
        # 新增要素
        "VIS": {
            "description": "Visibility",
            "grib_codes": {"value": "VIS"},
            "requires_uv": False,
            "output_vars": ["vis"],
            "units": {"vis": "m"},
            "micaps_type": "scalar",
            "micaps_element": "VIS"
        },
        "TCC": {
            "description": "Total cloud cover",
            "grib_codes": {"value": "TCC"},
            "requires_uv": False,
            "output_vars": ["tcc"],
            "units": {"tcc": "fraction"},
            "micaps_type": "scalar",
            "micaps_element": "TCC"
        }
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


# ================= MICAPS4写入器 =================
class MICAPS4Writer:
    """MICAPS第4类数据格式写入器"""
    
    @staticmethod
    def create_micaps4_filename(base_time: datetime, forecast_hour: int,
                               element: str = "VIS",
                               timezone_shift: timedelta = None) -> str:
        """
        创建MICAPS4格式文件名（月日时分.时效）
        
        Parameters
        ----------
        base_time : datetime
            预报起始时间（UTC）
        forecast_hour : int
            预报时效（小时）
        element : str
            要素名称
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
    def write_micaps4_scalar_file(data: np.ndarray, 
                                 lats: np.ndarray, lons: np.ndarray,
                                 base_time: datetime, forecast_hour: int,
                                 output_path: str, element: str = "VIS",
                                 model_name: str = "ECMWF",
                                 level: float = 0.0,
                                 description: str = None,
                                 timezone_shift: timedelta = None) -> bool:
        """
        写入MICAPS4标量格式文件（type=4）- 标准网格数据
        
        Parameters
        ----------
        data : np.ndarray
            2D数据数组
        lats : np.ndarray
            纬度数组（升序）
        lons : np.ndarray
            经度数组（升序）
        base_time : datetime
            预报起始时间（UTC）
        forecast_hour : int
            预报时效（小时）
        output_path : str
            输出文件路径
        element : str
            要素名称
        model_name : str
            模式名称
        level : float
            层次/高度
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
            
            # 设置默认参数
            if description is None:
                element_config = Config.ELEMENTS.get(element, {})
                description = element_config.get("description", element)
            
            # 限制描述长度
            if len(description) > 30:
                description = description[:30]
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 写入二进制文件
            with open(output_path, 'wb') as f:
                # ============= 1. discriminator =============
                f.write(b'mdfs')
                
                # ============= 2. type: 4为网格标量数据 =============
                data_type = 4  # 网格标量数据
                f.write(np.int16(data_type).tobytes())
                
                # ============= 3. modelName =============
                model_name_bytes = model_name.upper().encode('ascii', 'ignore')[:20]
                if len(model_name_bytes) < 20:
                    model_name_bytes += b'\x00' * (20 - len(model_name_bytes))
                f.write(model_name_bytes)
                
                # ============= 4. element =============
                element_std = element.upper()[:50]
                element_bytes = element_std.encode('ascii', 'ignore')[:50]
                if len(element_bytes) < 50:
                    element_bytes += b'\x00' * (50 - len(element_bytes))
                f.write(element_bytes)
                
                # ============= 5. description =============
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
                # 根据数据类型设置合适的等值线参数
                if element == "VIS":
                    isoline_start = 0.0
                    isoline_end = 10000.0
                    isoline_space = 1000.0
                elif element == "TCC":
                    isoline_start = 0.0
                    isoline_end = 1.0
                    isoline_space = 0.1
                elif element == "TEM":
                    isoline_start = -30.0
                    isoline_end = 40.0
                    isoline_space = 2.0
                else:
                    isoline_start = 0.0
                    isoline_end = 0.0
                    isoline_space = 0.0
                
                f.write(np.float32(isoline_start).tobytes())
                f.write(np.float32(isoline_end).tobytes())
                f.write(np.float32(isoline_space).tobytes())
                
                # ============= 24. Extent =============
                f.write(b'\x00' * 100)
                
                # ============= 数据区 =============
                # 处理NaN值（MICAPS使用9999.0表示缺测值）
                data_clean = data.copy()
                nan_mask = np.isnan(data_clean)
                
                if np.any(nan_mask):
                    data_clean[nan_mask] = 9999.0
                
                # 展平数据（按行优先顺序）
                data_flat = data_clean.ravel(order='C').astype(np.float32)
                
                # 写入数据
                f.write(data_flat.tobytes())
            
            file_size_kb = os.path.getsize(output_path) / 1024
            print(f"✓ MICAPS4标量文件已生成: {output_path}")
            print(f"  数据类型: 标量 (type=4)")
            print(f"  要素: {element}")
            print(f"  模式: {model_name}")
            print(f"  起报: {local_time.strftime('%Y-%m-%d %H:%M')} (UTC+8)")
            print(f"  时效: {forecast_hour:03d}小时")
            print(f"  层次: {level}")
            print(f"  网格: {n_lon}x{n_lat} ({start_lon:.2f}-{end_lon:.2f}E, {start_lat:.2f}-{end_lat:.2f}N)")
            print(f"  数据点: {data_flat.size} 个")
            print(f"  文件大小: {file_size_kb:.1f} KB")
            
            return True
            
        except Exception as e:
            print(f"✗ 写入MICAPS4标量文件失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


# ================= 处理类 =================
class ECProcessorSimple:
    """ECMWF数据处理器 - 简化版（类似小代码的思路）"""
    
    def __init__(self, save_micaps4: bool = False, micaps_output_dir: str = None):
        """
        初始化处理器
        
        Parameters
        ----------
        save_micaps4 : bool
            是否保存MICAPS4格式
        micaps_output_dir : str
            MICAPS4输出目录
        """
        self.save_micaps4 = save_micaps4
        self.micaps_output_dir = micaps_output_dir
        
        # 预计算目标网格
        self.lat_dst, self.lon_dst = Config.get_target_grid()
        self.n_lat_dst, self.n_lon_dst = self.lat_dst.size, self.lon_dst.size
    
    def process_element(self, element: str, input_file: str, output_file: str = None,
                       skip_existing: bool = True, save_micaps4: bool = None,
                       micaps_output_dir: str = None) -> Tuple[bool, str, List[str]]:
        """
        处理单个要素数据（简化接口）
        
        Args:
            element: 要素名称（VIS, TCC, TEM等）
            input_file: 输入GRIB文件路径
            output_file: 输出NC文件路径（可选）
            skip_existing: 是否跳过已存在的文件
            save_micaps4: 是否保存MICAPS4格式
            micaps_output_dir: MICAPS4输出目录
            
        Returns:
            tuple: (是否成功, NC文件路径, MICAPS4文件列表)
        """
        try:
            print(f"处理{Config.ELEMENTS[element]['description']}数据: {os.path.basename(input_file)}")
            
            # 确定是否保存MICAPS4
            if save_micaps4 is None:
                save_micaps4 = self.save_micaps4
            
            # 确定MICAPS4输出目录
            if micaps_output_dir is None:
                micaps_output_dir = self.micaps_output_dir
            if micaps_output_dir is None and output_file:
                micaps_output_dir = os.path.dirname(output_file)
            
            # 获取基准时间
            base_time = Config.parse_time_from_filename(input_file)
            if base_time is None:
                print("无法解析基准时间")
                return False, None, []
            
            # 确定输出路径
            if output_file is None:
                # 自动生成输出文件名
                bjt_time_str = Config.utc_to_bjt_str(base_time)
                output_filename = Config.OUTPUT_FILENAME_FORMAT.format(
                    element=element.lower(), time_str=bjt_time_str
                )
                output_file = os.path.join(os.path.dirname(input_file), output_filename)
            
            if skip_existing and os.path.exists(output_file):
                print(f"文件已存在: {output_file}")
                return True, output_file, []
            
            # 处理数据
            success, result_data, times_bjt = self._process_scalar_data(
                element, input_file, base_time, output_file
            )
            
            micaps4_files = []
            if success:
                file_size_mb = os.path.getsize(output_file) / 1024 / 1024
                print(f"✓ NetCDF处理完成: {output_file}")
                print(f"  文件大小: {file_size_mb:.1f} MB")
                
                # 保存MICAPS4格式
                if save_micaps4 and result_data is not None and micaps_output_dir:
                    micaps4_files = self._save_micaps4_files(
                        element, result_data, base_time, micaps_output_dir
                    )
            
            return success, output_file, micaps4_files
            
        except Exception as e:
            print(f"处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, None, []
    
    def _process_scalar_data(self, element: str, input_file: str,
                            base_time_utc: datetime, output_path: str) -> Tuple[bool, np.ndarray, List[datetime]]:
        """处理标量数据（VIS, TCC, TEM等）"""
        try:
            # 获取要素配置
            element_config = Config.ELEMENTS[element]
            
            # 1. 读取数据
            print("读取数据...")
            start_time = time.time()
            
            with pygrib.open(input_file) as grbs:
                # 获取所有时次
                steps = sorted({msg.step for msg in grbs})
                
                if not steps:
                    print("没有预报时次")
                    return False, None, None
                
                # 读取第一个消息获取网格信息
                msg = grbs.select(step=steps[0])[0]
                data, lats, lons = msg.data(
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
                    lat_src = lat_src[::-1]  # 反转成升序
                
                # 读取所有数据
                data_cube = []
                for step in steps:
                    msg = grbs.select(step=step)[0]
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
                    
                    data_cube.append(data)
                
                data_cube = np.array(data_cube, dtype=np.float32)
            
            read_time = time.time() - start_time
            print(f"数据读取完成: {read_time:.1f}秒, 时次: {len(steps)}")
            
            # 2. 空间插值
            print("空间插值...")
            start_time = time.time()
            
            # 准备目标网格点
            lon2d, lat2d = np.meshgrid(self.lon_dst, self.lat_dst)
            points = np.column_stack([lat2d.ravel(), lon2d.ravel()])
            
            # 创建插值器
            interp_func = RegularGridInterpolator(
                (lat_src, lon_src),
                data_cube[0],
                bounds_error=False,
                fill_value=np.nan
            )
            
            # 批量插值
            data_interp = np.empty((len(steps), self.n_lat_dst, self.n_lon_dst), dtype=np.float32)
            
            for i in range(len(steps)):
                interp_func.values = data_cube[i]
                data_interp[i] = interp_func(points).reshape(self.n_lat_dst, self.n_lon_dst)
            
            interp_time = time.time() - start_time
            print(f"空间插值完成: {interp_time:.1f}秒")
            
            # 3. 时间插值（分段处理）
            print("时间插值...")
            start_time = time.time()
            
            # 根据时效分两段：0-72小时(3h间隔)和73-240小时(6h间隔)
            steps_3h = [step for step in steps if step <= 72]
            steps_6h = [step for step in steps if step > 72 and step <= 240]
            
            # 分段处理
            data_1h_segments = []
            hour_segments = []
            
            # 第一段: 0-72小时 (3小时间隔)
            if steps_3h:
                indices_3h = [i for i, step in enumerate(steps) if step in steps_3h]
                data_3h = data_interp[indices_3h]
                
                hours_3h = steps_3h
                hours_new_3h = np.arange(0, 72 + 1, 1.0)
                hours_new_3h = hours_new_3h[hours_new_3h <= 72]
                
                f_data_3h = interp1d(hours_3h, data_3h, axis=0, kind='linear', fill_value='extrapolate')
                data_1h_3h = f_data_3h(hours_new_3h)
                
                data_1h_segments.append(data_1h_3h)
                hour_segments.append(hours_new_3h)
                
                print(f"  0-72小时: {len(hours_3h)}个原始时次 → {len(hours_new_3h)}个1小时间隔时次")
            
            # 第二段: 72-240小时 (6小时间隔)
            if steps_6h:
                indices_6h = [i for i, step in enumerate(steps) if step in steps_6h]
                data_6h = data_interp[indices_6h]
                
                hours_6h = steps_6h
                max_forecast_hour = max(steps_6h)
                start_hour_6h = 73
                end_hour_6h = min(max_forecast_hour, 240)
                
                if start_hour_6h <= end_hour_6h:
                    hours_new_6h = np.arange(start_hour_6h, end_hour_6h + 1, 1.0)
                    
                    f_data_6h = interp1d(hours_6h, data_6h, axis=0, kind='linear', fill_value='extrapolate')
                    data_1h_6h = f_data_6h(hours_new_6h)
                    
                    data_1h_segments.append(data_1h_6h)
                    hour_segments.append(hours_new_6h)
                    
                    print(f"  73-{end_hour_6h}小时: {len(hours_6h)}个原始时次 → {len(hours_new_6h)}个1小时间隔时次")
                else:
                    print(f"  72-240小时段无有效数据")
            
            # 合并所有分段
            if data_1h_segments:
                data_1h = np.concatenate(data_1h_segments, axis=0)
                hours_new = np.concatenate(hour_segments, axis=0)
                
                # 生成北京时时间序列
                times_bjt = [base_time_utc + timedelta(hours=float(h)) + Config.TIMEZONE_SHIFT 
                            for h in hours_new]
                
                time_interp_time = time.time() - start_time
                print(f"时间插值完成: {time_interp_time:.1f}秒, 总时次: {len(times_bjt)}")
            else:
                print("时间插值失败：没有生成有效数据")
                return False, None, None
            
            # 4. 写入NetCDF
            print("写入NetCDF文件...")
            start_time = time.time()
            
            # 创建临时文件路径
            temp_path = output_path + '.tmp'
            
            with Dataset(temp_path, 'w') as nc:
                # 创建维度
                nc.createDimension('time', len(times_bjt))
                nc.createDimension('lat', self.n_lat_dst)
                nc.createDimension('lon', self.n_lon_dst)
                
                # 坐标变量
                time_var = nc.createVariable('time', 'i4', ('time',))
                lat_var = nc.createVariable('lat', 'f4', ('lat',))
                lon_var = nc.createVariable('lon', 'f4', ('lon',))
                
                # 数据变量
                var_name = element_config["output_vars"][0]
                data_var = nc.createVariable(var_name, 'f4', ('time', 'lat', 'lon'),
                                            zlib=True, complevel=1, fill_value=-9999.0)
                
                # 设置坐标
                lat_var[:] = self.lat_dst
                lat_var.units = 'degrees_north'
                lat_var.long_name = 'latitude (south to north)'
                
                lon_var[:] = self.lon_dst
                lon_var.units = 'degrees_east'
                
                # 设置时间
                time_var.units = 'hours since 1970-01-01 00:00:00'
                time_var.calendar = 'gregorian'
                time_var.time_zone = 'UTC+8'
                time_var[:] = date2num(times_bjt, time_var.units, time_var.calendar)
                
                # 设置数据（处理NaN值）
                output_data = data_1h.copy()
                nan_mask = np.isnan(output_data)
                if np.any(nan_mask):
                    output_data[nan_mask] = -9999.0
                
                data_var[:] = output_data
                data_var.units = element_config["units"][var_name]
                data_var.long_name = element_config["description"]
                data_var.missing_value = -9999.0
                
                # 添加全局属性
                nc.title = f'ECMWF {element_config["description"]} (Beijing Time)'
                nc.source = 'ECMWF Forecast'
                nc.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                nc.forecast_start_time_utc = base_time_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
                nc.forecast_start_time_bjt = Config.utc_to_bjt_str(base_time_utc)
                nc.latitude_order = 'south to north (ascending)'
                nc.time_interpolation = '分段线性插值: 0-72小时(3h→1h), 73-240小时(6h→1h)'
            
            # 重命名临时文件为最终文件
            os.rename(temp_path, output_path)
            
            write_time = time.time() - start_time
            print(f"NetCDF文件写入完成: {write_time:.1f}秒")
            
            return True, data_1h, times_bjt
            
        except Exception as e:
            print(f"处理标量数据失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, None, None
    
    def _save_micaps4_files(self, element: str, result_data: np.ndarray,
                           base_time_utc: datetime, output_dir: str) -> List[str]:
        """
        保存MICAPS4格式文件（按要素创建子文件夹）
        
        Parameters
        ----------
        element : str
            要素名称
        result_data : np.ndarray
            处理结果数据（时间, 纬度, 经度）
        base_time_utc : datetime
            UTC基准时间
        output_dir : str
            输出目录
            
        Returns
        -------
        List[str]
            MICAPS4文件路径列表
        """
        if not output_dir:
            print("未指定MICAPS4输出目录，跳过保存")
            return []
        
        print(f"开始保存MICAPS4格式文件: {element}")
        
        # 获取要素配置
        element_config = Config.ELEMENTS.get(element, {})
        element_name = element_config.get("micaps_element", element)
        
        # 创建要素子文件夹
        element_dir = os.path.join(output_dir, element_name)
        os.makedirs(element_dir, exist_ok=True)
        print(f"MICAPS4文件将保存到: {element_dir}")
        
        micaps4_files = []
        
        start_time = time.time()
        
        # 逐个时次保存MICAPS4文件
        for i in range(result_data.shape[0]):
            forecast_hour = i
            
            # 生成文件名：月日时分.时效
            filename = MICAPS4Writer.create_micaps4_filename(
                base_time=base_time_utc,
                forecast_hour=forecast_hour,
                element=element_name,
                timezone_shift=Config.TIMEZONE_SHIFT
            )
            output_path = os.path.join(element_dir, filename)
            
            # 跳过全为NaN的数据
            if np.all(np.isnan(result_data[i])):
                continue
            
            # 设置适当的层次值
            if element == "VIS":
                level = 10.0  # 10米高度
            elif element == "TCC":
                level = 0.0  # 地面层
            elif element == "TEM":
                level = 2.0  # 2米高度
            else:
                level = 0.0
            
            success = MICAPS4Writer.write_micaps4_scalar_file(
                data=result_data[i],
                lats=self.lat_dst, lons=self.lon_dst,
                base_time=base_time_utc,
                forecast_hour=forecast_hour,
                output_path=output_path,
                element=element_name,
                model_name="ECMWF",
                level=level,
                description=element_config.get("description", element)
            )
            
            if success:
                micaps4_files.append(output_path)
            
            # 每处理50个时次打印一次进度
            if (i + 1) % 50 == 0 or i == result_data.shape[0] - 1:
                print(f"  进度: {i+1}/{result_data.shape[0]} 个时次")
        
        total_time = time.time() - start_time
        total_files = len(micaps4_files)
        print(f"MICAPS4文件保存完成: 共{total_files}个文件, 耗时: {total_time:.1f}秒")
        print(f"文件保存目录: {element_dir}")
        
        return micaps4_files


# ================= 命令行接口 =================
def main():
    parser = argparse.ArgumentParser(description='ECMWF VIS和TCC数据处理 - 支持MICAPS4输出')
    
    # 基本参数
    parser.add_argument('--element', type=str, required=True, 
                       choices=['VIS', 'TCC', 'TEM', 'PRS', 'DPT', 'GUST', 'RH', 'WIND'],
                       help='要素名称：VIS(能见度), TCC(总云量), TEM(气温)等')
    parser.add_argument('--file', type=str, required=True, help='输入GRIB文件路径')
    parser.add_argument('--output', type=str, help='输出NC文件路径（可选）')
    parser.add_argument('--output-dir', type=str, help='输出目录（可选，与--output互斥）')
    
    # MICAPS4参数
    parser.add_argument('--save-micaps4', action='store_true', default=False,
                       help='保存MICAPS4格式文件')
    parser.add_argument('--micaps-output-dir', type=str, default=None,
                       help='MICAPS4输出目录（如为None则使用NC输出目录）')
    
    # 处理选项
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='跳过已存在的输出文件')
    parser.add_argument('--no-skip-existing', action='store_false', dest='skip_existing',
                       help='不跳过已存在的输出文件')
    
    args = parser.parse_args()
    
    # 检查WIND要素的特殊要求
    if args.element == "WIND":
        print("警告：WIND要素需要U和V分量文件，当前版本仅支持标量要素")
        print("请使用专用的风场处理程序")
        return
    
    # 创建处理器
    processor = ECProcessorSimple(
        save_micaps4=args.save_micaps4,
        micaps_output_dir=args.micaps_output_dir
    )
    
    # 确定输出路径
    output_path = args.output
    
    # 如果指定了输出目录但没有指定输出文件
    if args.output_dir and not args.output:
        # 获取基准时间
        base_time = Config.parse_time_from_filename(args.file)
        if base_time:
            bjt_time_str = Config.utc_to_bjt_str(base_time)
            output_filename = Config.OUTPUT_FILENAME_FORMAT.format(
                element=args.element.lower(), time_str=bjt_time_str
            )
            output_path = os.path.join(args.output_dir, output_filename)
        else:
            print("无法解析基准时间，无法自动生成输出文件名")
            return
    
    print(f"{'='*60}")
    print(f"处理要素: {args.element}")
    print(f"输入文件: {args.file}")
    print(f"输出文件: {output_path}")
    print(f"保存MICAPS4: {'是' if args.save_micaps4 else '否'}")
    if args.save_micaps4:
        micaps_dir = args.micaps_output_dir or os.path.dirname(output_path) or "."
        print(f"MICAPS4输出目录: {micaps_dir}")
    print(f"{'='*60}")
    
    success, output_file, micaps_files = processor.process_element(
        element=args.element,
        input_file=args.file,
        output_file=output_path,
        skip_existing=args.skip_existing,
        save_micaps4=args.save_micaps4,
        micaps_output_dir=args.micaps_output_dir
    )
    
    print(f"\n{'='*60}")
    if success:
        print(f"✓ 处理成功")
        print(f"  NetCDF文件: {output_file}")
        if micaps_files:
            print(f"  MICAPS4文件: {len(micaps_files)} 个")
            if len(micaps_files) <= 10:
                for file in micaps_files[:5]:
                    print(f"    - {os.path.basename(file)}")
                if len(micaps_files) > 5:
                    print(f"    ... 等{len(micaps_files)}个文件")
            else:
                print(f"    - {os.path.basename(micaps_files[0])}")
                print(f"    - {os.path.basename(micaps_files[1])}")
                print(f"    - ... 等{len(micaps_files)-2}个文件")
                print(f"    - {os.path.basename(micaps_files[-1])}")
    else:
        print(f"✗ 处理失败")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


'''
使用示例：
# 处理能见度，输出MICAPS4格式
python EC_V3_VIS_TCC_20260206.py --element VIS --file /home/youqi/FWZX_forecast_DATA/data_demo/ECMFC1D_VIS_1_2026013100_GLB_1.grib1 --output-dir /home/youqi/FWZX_forecast_DATA/output --save-micaps4 --micaps-output-dir /home/youqi/FWZX_forecast_DATA/tonst


# 处理总云量，输出MICAPS4格式，指定MICAPS4输出目录
python EC_V3_VIS_TCC_20260206.py --element TCC --file /path/to/ECMFC1D_TCC_1_2026013100_GLB_1.grib1 --output-dir /path/to/output --save-micaps4 --micaps-output-dir /path/to/micaps_output

# 处理气温，不输出MICAPS4格式
python EC_V3_VIS_TCC_20260206.py --element TEM --file /path/to/ECMFC1D_TEM_1_2026010100_GLB_1.grib1 --output-dir /path/to/output

# 指定输出文件
python EC_V3_VIS_TCC_20260206.py --element VIS --file input.grib --output ./output/custom_name.nc --save-micaps4

输出文件结构：
/path/to/micaps_output/
├── VIS/           # 能见度文件夹
│   ├── 010100.000    # 1月1日00时，0小时预报
│   ├── 010100.001    # 1月1日00时，1小时预报
│   ├── 010100.002    # 1月1日00时，2小时预报
│   └── ...
├── TCC/           # 总云量文件夹
│   ├── 010100.000
│   ├── 010100.001
│   ├── 010100.002
│   └── ...
├── TEM/           # 气温文件夹
│   ├── 010100.000
│   ├── 010100.001
│   ├── 010100.002
│   └── ...
└── ...

文件名说明：
010100.000 = 1月1日00时预报，0小时时效
010100.001 = 1月1日00时预报，1小时时效
011200.024 = 1月12日00时预报，24小时时效
'''