
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECMWF降水数据处理模块 - 支持MICAPS4格式输出
"""

import os
import sys
import numpy as np
from datetime import datetime, timedelta
import pygrib
from scipy.interpolate import RegularGridInterpolator
from netCDF4 import Dataset, date2num
import time
from typing import Tuple, List, Dict, Optional
import warnings

# 导入配置文件
try:
    # 尝试从主文件导入
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from EC_V3_NC_MICAPS_tmp临时文件_20260205 import Config, MICAPS4Writer
except ImportError:
    print("警告: 无法从主文件导入，使用独立配置")
    
    # 独立配置
    class Config:
        REGION = {"lon_w": 110.0, "lon_e": 125.0, "lat_s": 34.0, "lat_n": 44.0}
        RESOLUTION = 0.01
        TIMEZONE_SHIFT = timedelta(hours=8)
        OUTPUT_FILENAME_FORMAT = "{element}_0p01_1h_BJT_{time_str}.nc"
        
        # 降水要素配置
        PRECIP_CONFIG = {
            "description": "Hourly precipitation",
            "grib_codes": {"value": "TP"},
            "requires_uv": False,
            "output_vars": ["precip"],
            "units": {"precip": "mm"}
        }
        
        @classmethod
        def get_target_grid(cls):
            lon_out = np.arange(cls.REGION["lon_w"], cls.REGION["lon_e"] + cls.RESOLUTION/2, cls.RESOLUTION)
            lat_out = np.arange(cls.REGION["lat_s"], cls.REGION["lat_n"] + cls.RESOLUTION/2, cls.RESOLUTION)
            return lat_out, lon_out
        
        @classmethod
        def utc_to_bjt_str(cls, utc_time):
            bjt_time = utc_time + cls.TIMEZONE_SHIFT
            return bjt_time.strftime("%Y%m%d%H")
        
        @classmethod
        def parse_time_from_filename(cls, filename):
            import re
            basename = os.path.basename(filename)
            match = re.search(r'(\d{10})', basename)
            if match:
                try:
                    return datetime.strptime(match.group(1), "%Y%m%d%H")
                except:
                    pass
            return None
    
    # MICAPS4写入器（简化版）
    class MICAPS4Writer:
        """MICAPS第4类数据格式写入器 - 二进制格式"""
        
        @staticmethod
        def create_micaps4_filename(base_time: datetime, forecast_hour: int,
                                   element: str = "PRECIP", model_name: str = "ECMWF",
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
                                     output_path: str, element: str = "PRECIP", 
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
                    element_std = "PRECIP"  # 降水要素名
                    element_bytes = element_std.encode('ascii', 'ignore')[:50]
                    if len(element_bytes) < 50:
                        element_bytes += b'\x00' * (50 - len(element_bytes))
                    f.write(element_bytes)
                    
                    # ============= 5. description =============
                    desc = description or "Hourly precipitation (mm)"
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
                
                file_size_kb = os.path.getsize(output_path) / 1024
                print(f"✓ MICAPS4降水文件已生成: {output_path}")
                print(f"  数据类型: 标量 (type=4)")
                print(f"  要素: {element_std}")
                print(f"  模式: {model_name}")
                print(f"  起报: {local_time.strftime('%Y-%m-%d %H:%M')} (UTC+8)")
                print(f"  时效: {forecast_hour:03d}小时")
                print(f"  层次: {level} hPa")
                print(f"  网格: {n_lon}x{n_lat} ({start_lon:.2f}-{end_lon:.2f}E, {start_lat:.2f}-{end_lat:.2f}N)")
                print(f"  数据点: {data_flat.size} 个")
                print(f"  文件大小: {file_size_kb:.1f} KB")
                
                return True
                
            except Exception as e:
                print(f"✗ 写入MICAPS4降水文件失败: {str(e)}")
                import traceback
                traceback.print_exc()
                return False


class PrecipitationProcessor:
    """ECMWF降水数据处理器 - 支持MICAPS4格式"""
    
    def __init__(self, save_micaps4: bool = False, micaps4_output_dir: str = None):
        # 预计算目标网格
        self.lat_dst, self.lon_dst = Config.get_target_grid()
        self.n_lat_dst, self.n_lon_dst = self.lat_dst.size, self.lon_dst.size
        
        # MICAPS4输出配置
        self.save_micaps4 = save_micaps4
        self.micaps4_output_dir = micaps4_output_dir
        
    def _safe_get_attr(self, msg, attr_name, default=None):
        try:
            return getattr(msg, attr_name, default)
        except (RuntimeError, KeyError):
            try:
                return msg[attr_name]
            except:
                return default
    
    def analyze_precipitation_structure(self, grib_file: str) -> Dict:
        print(f"分析降水文件结构: {os.path.basename(grib_file)}")
        
        try:
            grbs = pygrib.open(grib_file)
            
            # 获取所有降水消息
            precip_msgs = []
            for msg in grbs:
                param_name = self._safe_get_attr(msg, 'parameterName', '')
                param_id = self._safe_get_attr(msg, 'paramId', -1)
                short_name = self._safe_get_attr(msg, 'shortName', '')
                
                if (param_name == 'Total precipitation' or 
                    param_id == 228 or 
                    short_name == 'tp'):
                    precip_msgs.append(msg)
            
            if not precip_msgs:
                print("文件中未找到降水数据")
                grbs.close()
                return {}
            
            # 获取预报时效并排序
            forecast_times = []
            sorted_msgs = []
            
            for msg in precip_msgs:
                forecast_time = self._safe_get_attr(msg, 'forecastTime')
                if forecast_time is None:
                    forecast_time = self._safe_get_attr(msg, 'endStep', 0)
                forecast_times.append(forecast_time)
            
            # 按时效排序
            msg_time_pairs = list(zip(forecast_times, precip_msgs))
            msg_time_pairs.sort(key=lambda x: x[0])
            
            sorted_forecast_times = [t for t, _ in msg_time_pairs]
            sorted_msgs = [m for _, m in msg_time_pairs]
            
            # 提取信息
            values_avg = []
            
            for msg in sorted_msgs:
                try:
                    data = msg.values
                    avg_value = np.mean(data)
                except:
                    avg_value = np.nan
                values_avg.append(avg_value)
            
            structure_info = {
                'forecast_times': sorted_forecast_times,
                'messages': sorted_msgs,
                'values_avg': values_avg,
                'n_messages': len(sorted_msgs)
            }
            
            grbs.close()
            return structure_info
            
        except Exception as e:
            print(f"分析失败: {str(e)}")
            if 'grbs' in locals():
                try:
                    grbs.close()
                except:
                    pass
            return {}
    
    def process_precipitation(self, input_file: str, output_file: str = None,
                            output_dir: str = None,  # 添加output_dir参数
                            skip_existing: bool = True, save_micaps4: bool = None,
                            micaps4_output_dir: str = None) -> Tuple[bool, str, Dict]:
        """处理降水数据
        
        Args:
            input_file: 输入GRIB文件路径
            output_file: 输出NC文件路径（可选，如未指定则自动生成）
            output_dir: 输出目录（可选，与output_file互斥）
            skip_existing: 是否跳过已存在的文件
            save_micaps4: 是否保存MICAPS4格式
            micaps4_output_dir: MICAPS4输出目录
            
        Returns:
            tuple: (是否成功, NetCDF输出文件路径, MICAPS4文件信息字典)
        """
        total_start = time.time()
        success = False
        output_path = None
        micaps4_files = {}
        
        try:
            print(f"处理降水数据: {os.path.basename(input_file)}")
            
            # 确定是否保存MICAPS4
            if save_micaps4 is None:
                save_micaps4 = self.save_micaps4
            
            # 确定MICAPS4输出目录
            if micaps4_output_dir is None:
                micaps4_output_dir = self.micaps4_output_dir
            
            # 分析结构
            structure_info = self.analyze_precipitation_structure(input_file)
            if not structure_info:
                return False, None, {}
            
            # 获取基准时间
            base_time = Config.parse_time_from_filename(input_file)
            if base_time is None:
                print("无法解析基准时间")
                return False, None, {}
            
            # 确定输出路径
            if output_file:
                output_path = output_file
            else:
                # 自动生成输出文件名
                bjt_time_str = Config.utc_to_bjt_str(base_time)
                output_filename = f"precip_0p01_1h_BJT_{bjt_time_str}.nc"
                
                if output_dir:
                    # 使用指定的输出目录
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, output_filename)
                else:
                    # 使用输入文件所在目录
                    output_path = os.path.join(os.path.dirname(input_file), output_filename)
            
            if skip_existing and os.path.exists(output_path):
                print(f"文件已存在: {output_path}")
                return True, output_path, {}
            
            # 处理数据
            result_data, times_bjt = self._process_precipitation_data(
                input_file, structure_info, base_time, output_path
            )
            
            if result_data is not None:
                success = True
                
                # 保存MICAPS4格式
                if success and save_micaps4:
                    micaps4_files = self._save_micaps4_files(
                        result_data, base_time, micaps4_output_dir or os.path.dirname(output_path)
                    )
            
            total_time = time.time() - total_start
            
            if success:
                file_size_mb = os.path.getsize(output_path) / 1024 / 1024
                print(f"✓ 处理完成: {output_path}")
                print(f"  文件大小: {file_size_mb:.1f} MB")
                print(f"  总耗时: {total_time:.1f}秒")
                
                if micaps4_files:
                    micaps4_count = len(micaps4_files.get("scalar", []))
                    print(f"  MICAPS4文件: {micaps4_count} 个")
                
                return True, output_path, micaps4_files
            else:
                print(f"✗ 处理失败")
                return False, output_path, micaps4_files
            
        except Exception as e:
            print(f"处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, output_path, micaps4_files
    
    def _process_precipitation_data(self, input_file: str, structure_info: Dict,
                                  base_time_utc: datetime, output_path: str) -> Tuple[Optional[Dict], Optional[List]]:
        try:
            # 1. 读取累积降水
            accum_data, lat_src, lon_src, forecast_times = self._read_accumulated_precipitation(
                structure_info
            )
            
            if accum_data is None:
                return None, None
            
            # 2. 空间插值
            accum_interp = self._spatial_interpolation(accum_data, lat_src, lon_src)
            
            # 3. 单位转换：米 → 毫米
            accum_interp = accum_interp * 1000.0
            
            # 4. 计算时段降水
            interval_precip = self._calculate_interval_precipitation(accum_interp, forecast_times)
            
            # 5. 降尺度到1小时
            hourly_precip = self._downscale_to_hourly(interval_precip, forecast_times)
            
            # 6. 生成时间序列
            times_bjt = self._generate_bjt_time_series(base_time_utc, hourly_precip.shape[0])
            
            # 7. 写入文件
            success = self._write_netcdf_output(hourly_precip, times_bjt, output_path, base_time_utc)
            
            if success:
                return {"precip": hourly_precip}, times_bjt
            else:
                return None, None
            
        except Exception as e:
            print(f"处理失败: {str(e)}")
            return None, None
    
    def _read_accumulated_precipitation(self, structure_info: Dict) -> Tuple:
        try:
            messages = structure_info['messages']
            forecast_times = structure_info['forecast_times']
            
            if not messages:
                return None, None, None, None
            
            # 读取第一个消息获取网格
            msg = messages[0]
            data, lats, lons = msg.data(
                lat1=Config.REGION["lat_s"],
                lat2=Config.REGION["lat_n"],
                lon1=Config.REGION["lon_w"],
                lon2=Config.REGION["lon_e"]
            )
            
            # 获取源网格
            lat_src = lats[:, 0]
            lon_src = lons[0, :]
            
            # 确保纬度升序
            if lat_src[-1] < lat_src[0]:
                lat_src = lat_src[::-1]
            
            n_lat, n_lon = len(lat_src), len(lon_src)
            n_times = len(messages)
            
            # 读取所有数据
            accum_data = np.empty((n_times, n_lat, n_lon), dtype=np.float32)
            
            for i, msg in enumerate(messages):
                data, lats, lons = msg.data(
                    lat1=Config.REGION["lat_s"],
                    lat2=Config.REGION["lat_n"],
                    lon1=Config.REGION["lon_w"],
                    lon2=Config.REGION["lon_e"]
                )
                
                # 确保方向一致
                if lat_src[-1] > lat_src[0] and lats[-1, 0] < lats[0, 0]:
                    data = data[::-1, :]
                
                accum_data[i] = data
            
            print(f"读取完成: {n_times}个时次, 网格: {n_lat}x{n_lon}")
            print(f"预报时效: {forecast_times}")
            
            return accum_data, lat_src, lon_src, forecast_times
            
        except Exception as e:
            print(f"读取失败: {str(e)}")
            return None, None, None, None
    
    def _spatial_interpolation(self, data: np.ndarray, lat_src: np.ndarray, 
                              lon_src: np.ndarray) -> np.ndarray:
        print("开始空间插值...")
        start_time = time.time()
        
        lon2d, lat2d = np.meshgrid(self.lon_dst, self.lat_dst)
        points = np.column_stack([lat2d.ravel(), lon2d.ravel()])
        
        n_times = data.shape[0]
        data_interp = np.empty((n_times, self.n_lat_dst, self.n_lon_dst), dtype=np.float32)
        
        for i in range(n_times):
            interp_func = RegularGridInterpolator(
                (lat_src, lon_src),
                data[i],
                bounds_error=False,
                fill_value=np.nan
            )
            
            interp_result = interp_func(points)
            data_interp[i] = interp_result.reshape(self.n_lat_dst, self.n_lon_dst)
        
        interp_time = time.time() - start_time
        print(f"空间插值完成: {interp_time:.1f}秒")
        
        return data_interp
    
    def _calculate_interval_precipitation(self, accum_data: np.ndarray, 
                                        forecast_times: List[int]) -> np.ndarray:
        """计算时段降水"""
        print("计算时段降水...")
        n_times = len(forecast_times)
        n_intervals = n_times - 1
        
        if n_intervals <= 0:
            return np.array([])
        
        interval_precip = np.empty((n_intervals, accum_data.shape[1], accum_data.shape[2]), dtype=np.float32)
        
        for i in range(n_intervals):
            # 时段降水 = 后一时刻累积 - 前一时刻累积
            interval_precip[i] = accum_data[i+1] - accum_data[i]
            
            # 检查负值（由于数值误差可能出现微小负值）
            negative_mask = interval_precip[i] < -0.001  # 允许微小负值
            negative_count = np.sum(negative_mask)
            if negative_count > 0:
                print(f"  时段{i}: 发现{negative_count}个负值，设为0")
                interval_precip[i][negative_mask] = 0.0
        
        print(f"时段降水计算完成: {n_intervals}个时段")
        return interval_precip
    
    def _downscale_to_hourly(self, interval_precip: np.ndarray, 
                            forecast_times: List[int]) -> np.ndarray:
        """降尺度到1小时 - 均匀分配"""
        print("降尺度到小时降水...")
        
        # 计算时段间隔
        intervals = []
        for i in range(1, len(forecast_times)):
            interval = forecast_times[i] - forecast_times[i-1]
            intervals.append(interval)
        
        total_hours = forecast_times[-1]
        
        n_lat, n_lon = interval_precip.shape[1], interval_precip.shape[2]
        hourly_precip = np.zeros((total_hours, n_lat, n_lon), dtype=np.float32)
        
        # 均匀分配到每小时
        for i, (interval, precip) in enumerate(zip(intervals, interval_precip)):
            start_time = forecast_times[i]
            
            if interval <= 0:
                continue
            
            # 均匀分配
            hourly_amount = precip / float(interval)
            
            # 分配到小时
            for hour_offset in range(interval):
                hour_idx = start_time + hour_offset
                if hour_idx < total_hours:
                    hourly_precip[hour_idx] = hourly_amount
        
        print(f"降尺度完成: {total_hours}小时")
        return hourly_precip
    
    def _generate_bjt_time_series(self, base_time_utc: datetime, n_hours: int) -> List[datetime]:
        times_bjt = []
        for hour in range(n_hours):
            utc_time = base_time_utc + timedelta(hours=hour)
            bjt_time = utc_time + Config.TIMEZONE_SHIFT
            times_bjt.append(bjt_time)
        
        return times_bjt
    
    def _write_netcdf_output(self, hourly_precip: np.ndarray, times_bjt: List[datetime],
                           output_path: str, base_time_utc: datetime) -> bool:
        try:
            print("写入NetCDF文件...")
            start_time = time.time()
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            temp_path = output_path + '.tmp'
            
            with Dataset(temp_path, 'w') as nc:
                # 维度
                nc.createDimension('time', len(times_bjt))
                nc.createDimension('lat', self.n_lat_dst)
                nc.createDimension('lon', self.n_lon_dst)
                
                # 坐标变量
                time_var = nc.createVariable('time', 'i4', ('time',))
                lat_var = nc.createVariable('lat', 'f4', ('lat',))
                lon_var = nc.createVariable('lon', 'f4', ('lon',))
                
                # 数据变量 - 在创建时设置fill_value
                precip_var = nc.createVariable('precip', 'f4', ('time', 'lat', 'lon'),
                                             zlib=True, complevel=1, fill_value=-9999.0)
                
                # 设置坐标
                lat_var[:] = self.lat_dst
                lat_var.units = 'degrees_north'
                lat_var.long_name = 'latitude (south to north)'
                
                lon_var[:] = self.lon_dst
                lon_var.units = 'degrees_east'
                lon_var.long_name = 'longitude'
                
                # 设置时间
                time_var.units = 'hours since 1970-01-01 00:00:00'
                time_var.calendar = 'gregorian'
                time_var.time_zone = 'UTC+8'
                time_var[:] = date2num(times_bjt, time_var.units, time_var.calendar)
                
                # 设置降水数据
                precip_data = hourly_precip.copy()
                nan_mask = np.isnan(precip_data)
                if np.any(nan_mask):
                    precip_data[nan_mask] = -9999.0
                
                precip_var[:] = precip_data
                precip_var.units = 'mm'
                precip_var.long_name = 'Hourly precipitation'
                precip_var.missing_value = -9999.0
                
                # 全局属性
                nc.title = 'ECMWF Hourly Precipitation (Beijing Time)'
                nc.source = 'ECMWF Forecast'
                nc.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                nc.forecast_start_time_utc = base_time_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
                nc.forecast_start_time_bjt = Config.utc_to_bjt_str(base_time_utc)
                nc.latitude_order = 'south to north (ascending)'
                nc.processing_method = '累积降水转换为小时降水，均匀分配'
            
            os.rename(temp_path, output_path)
            
            write_time = time.time() - start_time
            file_size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"NetCDF写入完成: {write_time:.1f}秒, 大小: {file_size_mb:.1f} MB")
            
            return True
            
        except Exception as e:
            print(f"写入失败: {str(e)}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            return False
    
    def _save_micaps4_files(self, result_data: Dict[str, np.ndarray],
                           base_time_utc: datetime, output_dir: str) -> Dict[str, List[str]]:
        """
        保存MICAPS4格式文件
        
        Parameters
        ----------
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
            print("未指定MICAPS4输出目录，跳过保存")
            return {}
        
        print("开始保存MICAPS4格式文件...")
        
        # 创建降水要素对应的子目录
        element_output_dir = os.path.join(output_dir, "PRECIP")
        os.makedirs(element_output_dir, exist_ok=True)
        
        micaps4_files = {"scalar": []}
        
        start_time = time.time()
        
        # 降水数据
        precip_data = result_data.get("precip")
        
        if precip_data is not None:
            for i in range(precip_data.shape[0]):
                forecast_hour = i
                
                # 生成文件名（使用新的命名规则）
                filename = MICAPS4Writer.create_micaps4_filename(
                    base_time=base_time_utc,
                    forecast_hour=forecast_hour,
                    element="PRECIP",
                    model_name="ECMWF"
                )
                output_path = os.path.join(element_output_dir, filename)
                
                if np.all(np.isnan(precip_data[i])):
                    continue
                
                success = MICAPS4Writer.write_micaps4_scalar_file(
                    data=precip_data[i], lats=self.lat_dst, lons=self.lon_dst,
                    base_time=base_time_utc, forecast_hour=forecast_hour,
                    output_path=output_path, element="PRECIP", model_name="ECMWF",
                    description="Hourly precipitation (mm)"
                )
                
                if success:
                    micaps4_files["scalar"].append(output_path)
        
        total_time = time.time() - start_time
        total_files = len(micaps4_files["scalar"])
        print(f"MICAPS4文件保存完成: 共{total_files}个文件, 耗时: {total_time:.1f}秒")
        print(f"输出目录: {element_output_dir}")
        
        return micaps4_files
    
    def batch_process_directory(self, input_dir: str, output_dir: str = None,
                              save_micaps4: bool = False, micaps4_output_dir: str = None,
                              skip_existing: bool = True) -> Dict[str, int]:
        """
        批量处理目录中的降水文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            save_micaps4: 是否保存MICAPS4格式
            micaps4_output_dir: MICAPS4输出目录
            skip_existing: 是否跳过已存在的文件
            
        Returns:
            处理统计信息
        """
        import glob
        
        stats = {'total': 0, 'success': 0, 'failed': 0, 'micaps4_files': 0}
        
        print(f"批量处理降水文件目录: {input_dir}")
        
        # 查找降水文件
        precip_patterns = [
            "*tp*.grib*", "*TP*.grib*", "*precip*.grib*", "*PRECIP*.grib*",
            "*total_precipitation*.grib*", "*TOTAL_PRECIPITATION*.grib*"
        ]
        
        all_files = []
        for pattern in precip_patterns:
            files = glob.glob(os.path.join(input_dir, pattern))
            all_files.extend(files)
        
        all_files = sorted(set(all_files))
        
        if not all_files:
            print(f"在目录 {input_dir} 中未找到降水文件")
            return stats
        
        print(f"找到 {len(all_files)} 个降水文件")
        
        total_start = time.time()
        
        for i, input_file in enumerate(all_files):
            stats['total'] += 1
            print(f"\n{'='*60}")
            print(f"处理文件 {i+1}/{len(all_files)}: {os.path.basename(input_file)}")
            
            # 处理文件 - 修正：传递正确的参数
            success, output_path, micaps4_files = self.process_precipitation(
                input_file=input_file,
                output_dir=output_dir,  # 传递output_dir而不是output_file
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
        
        total_time = time.time() - total_start
        
        print(f"\n{'='*80}")
        print("批量处理完成！")
        print(f"总文件数: {stats['total']}")
        print(f"成功: {stats['success']}")
        print(f"失败: {stats['failed']}")
        print(f"总MICAPS4文件数: {stats['micaps4_files']}")
        print(f"总耗时: {total_time:.1f}秒")
        if stats['total'] > 0:
            print(f"平均时间: {total_time/stats['total']:.1f}秒/文件")
        print(f"{'='*80}")
        
        return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ECMWF降水处理 - 支持MICAPS4格式输出')
    parser.add_argument('--file', type=str, help='输入文件')
    parser.add_argument('--output', type=str, help='输出文件路径（可选）')
    parser.add_argument('--output-dir', type=str, help='输出目录（可选）')
    
    # MICAPS4参数
    parser.add_argument('--save-micaps4', action='store_true', default=False,
                       help='保存MICAPS4格式文件')
    parser.add_argument('--micaps4-output-dir', type=str, default=None,
                       help='MICAPS4输出目录')
    
    # 批量处理
    parser.add_argument('--input-dir', type=str, help='输入目录（批量处理）')
    
    # 处理选项
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='跳过已存在的输出文件')
    parser.add_argument('--no-skip-existing', action='store_false', dest='skip_existing',
                       help='不跳过已存在的输出文件')
    
    # 配置参数
    parser.add_argument('--lon-west', type=float, default=110.0,
                       help='区域西边界经度')
    parser.add_argument('--lon-east', type=float, default=125.0,
                       help='区域东边界经度')
    parser.add_argument('--lat-south', type=float, default=34.0,
                       help='区域南边界纬度')
    parser.add_argument('--lat-north', type=float, default=44.0,
                       help='区域北边界纬度')
    parser.add_argument('--resolution', type=float, default=0.01,
                       help='输出分辨率（度）')
    
    # 测试模式
    parser.add_argument('--test', action='store_true', help='测试模式，仅分析文件结构')
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.file and not args.input_dir:
        print("错误: 必须提供 --file 或 --input-dir 参数")
        parser.print_help()
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
    
    # 设置Config（如果可导入）
    try:
        Config.REGION = config["REGION"]
        Config.RESOLUTION = config["RESOLUTION"]
    except:
        pass
    
    print(f"{'='*80}")
    print("ECMWF降水数据处理器")
    print(f"区域范围: {args.lon_west}E - {args.lon_east}E, {args.lat_south}N - {args.lat_north}N")
    print(f"分辨率: {args.resolution}度")
    print(f"保存MICAPS4格式: {'是' if args.save_micaps4 else '否'}")
    print(f"{'='*80}")
    
    processor = PrecipitationProcessor(
        save_micaps4=args.save_micaps4,
        micaps4_output_dir=args.micaps4_output_dir
    )
    
    try:
        if args.test:
            # 测试模式，仅分析文件结构
            if args.file:
                info = processor.analyze_precipitation_structure(args.file)
                if info:
                    print(f"\n文件结构分析:")
                    print(f"  消息数量: {info['n_messages']}")
                    print(f"  预报时效: {info['forecast_times']}")
                    print(f"  平均降水值 (m): {[f'{v:.6f}' for v in info['values_avg']]}")
                    print(f"  最大时效: {max(info['forecast_times'])} 小时")
            else:
                print("测试模式需要指定 --file 参数")
        
        elif args.input_dir:
            # 批量处理目录
            stats = processor.batch_process_directory(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                save_micaps4=args.save_micaps4,
                micaps4_output_dir=args.micaps4_output_dir,
                skip_existing=args.skip_existing
            )
            
            print(f"\n处理统计:")
            print(f"  总文件: {stats['total']}")
            print(f"  成功: {stats['success']}")
            print(f"  失败: {stats['failed']}")
            if args.save_micaps4:
                print(f"  MICAPS4文件: {stats['micaps4_files']}")
        
        else:
            # 处理单个文件
            success, output_file, micaps4_files = processor.process_precipitation(
                input_file=args.file,
                output_file=args.output,
                output_dir=args.output_dir,
                skip_existing=args.skip_existing,
                save_micaps4=args.save_micaps4,
                micaps4_output_dir=args.micaps4_output_dir
            )
            
            if success:
                print(f"\n✓ 成功输出:")
                print(f"  NetCDF文件: {output_file}")
                if micaps4_files:
                    micaps4_count = sum(len(files) for files in micaps4_files.values())
                    print(f"  MICAPS4文件: {micaps4_count} 个")
                    micaps4_dir = args.micaps4_output_dir or args.output_dir or os.path.dirname(output_file)
                    print(f"  输出目录: {os.path.join(micaps4_dir, 'PRECIP')}")
            else:
                print(f"\n✗ 处理失败")
                return 1
    
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    sys.exit(main())

'''
使用示例：

# 单个文件处理，不保存MICAPS4
python EC_V3_PRE_20260205.py --file /path/to/precip.grib

# 单个文件处理，保存MICAPS4
python EC_V3_PRE_20260205.py --file /path/to/precip.grib --save-micaps4 --micaps4-output-dir /path/to/micaps_output

# 批量处理目录
python EC_V3_PRE_20260205.py --input-dir /home/youqi/FWZX_forecast_DATA/data_demo --output-dir /home/youqi/FWZX_forecast_DATA/output --save-micaps4 --micaps4-output-dir /home/youqi/FWZX_forecast_DATA/tonst

# 测试模式，仅分析文件结构
python EC_V3_PRE_20260205.py --file /path/to/precip.grib --test

输出文件结构：
NetCDF文件: /path/to/output/precip_0p01_1h_BJT_2026020200.nc
MICAPS4文件: /path/to/micaps_output/PRECIP/26020200.000, 26020200.001, ...
'''
