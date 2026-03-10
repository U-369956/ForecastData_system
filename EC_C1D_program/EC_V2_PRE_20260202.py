#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECMWF降水数据处理模块 - 简化版
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

# 导入配置文件
try:
    from EC_V2_NC_MICAPS_tmp临时文件 import Config, MeteorologicalCalculator, MICAPS4Writer
except ImportError:
    # 如果无法导入，创建简化的配置
    class Config:
        REGION = {"lon_w": 110.0, "lon_e": 125.0, "lat_s": 34.0, "lat_n": 44.0}
        RESOLUTION = 0.01
        TIMEZONE_SHIFT = timedelta(hours=8)
        
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


class PrecipitationProcessor:
    """ECMWF降水数据处理器 - 简化版"""
    
    def __init__(self):
        # 预计算目标网格
        self.lat_dst, self.lon_dst = Config.get_target_grid()
        self.n_lat_dst, self.n_lon_dst = self.lat_dst.size, self.lon_dst.size
        
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
                
                if param_name == 'Total precipitation' or param_id == 228:
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
                'values_avg': values_avg
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
                            skip_existing: bool = True) -> Tuple[bool, str]:
        """处理降水数据
        
        Args:
            input_file: 输入GRIB文件路径
            output_file: 输出NC文件路径（可选，如未指定则自动生成）
            skip_existing: 是否跳过已存在的文件
        """
        try:
            print(f"处理降水数据: {os.path.basename(input_file)}")
            
            # 分析结构
            structure_info = self.analyze_precipitation_structure(input_file)
            if not structure_info:
                return False, None
            
            # 获取基准时间
            base_time = Config.parse_time_from_filename(input_file)
            if base_time is None:
                print("无法解析基准时间")
                return False, None
            
            # 确定输出路径
            if output_file is None:
                # 自动生成输出文件名
                bjt_time_str = Config.utc_to_bjt_str(base_time)
                output_filename = f"precip_0p01_1h_BJT_{bjt_time_str}.nc"
                output_file = os.path.join(os.path.dirname(input_file), output_filename)
            
            if skip_existing and os.path.exists(output_file):
                print(f"文件已存在: {output_file}")
                return True, output_file
            
            # 处理数据
            success = self._process_precipitation_data(
                input_file, structure_info, base_time, output_file
            )
            
            if success:
                file_size_mb = os.path.getsize(output_file) / 1024 / 1024
                print(f"✓ 处理完成: {output_file}")
                print(f"  文件大小: {file_size_mb:.1f} MB")
                return True, output_file
            else:
                return False, output_file
            
        except Exception as e:
            print(f"处理失败: {str(e)}")
            return False, None
    
    def _process_precipitation_data(self, input_file: str, structure_info: Dict,
                                  base_time_utc: datetime, output_path: str) -> bool:
        try:
            # 1. 读取累积降水
            accum_data, lat_src, lon_src, forecast_times = self._read_accumulated_precipitation(
                structure_info
            )
            
            if accum_data is None:
                return False
            
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
            
            return success
            
        except Exception as e:
            print(f"处理失败: {str(e)}")
            return False
    
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
            
            return accum_data, lat_src, lon_src, forecast_times
            
        except Exception as e:
            print(f"读取失败: {str(e)}")
            return None, None, None, None
    
    def _spatial_interpolation(self, data: np.ndarray, lat_src: np.ndarray, 
                              lon_src: np.ndarray) -> np.ndarray:
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
        
        return data_interp
    
    def _calculate_interval_precipitation(self, accum_data: np.ndarray, 
                                        forecast_times: List[int]) -> np.ndarray:
        """计算时段降水"""
        n_times = len(forecast_times)
        n_intervals = n_times - 1
        
        if n_intervals <= 0:
            return np.array([])
        
        interval_precip = np.empty((n_intervals, accum_data.shape[1], accum_data.shape[2]), dtype=np.float32)
        
        for i in range(n_intervals):
            # 时段降水 = 后一时刻累积 - 前一时刻累积
            interval_precip[i] = accum_data[i+1] - accum_data[i]
            
            # 检查负值
            negative_mask = interval_precip[i] < 0
            negative_count = np.sum(negative_mask)
            if negative_count > 0:
                interval_precip[i][negative_mask] = 0.0
        
        return interval_precip
    
    def _downscale_to_hourly(self, interval_precip: np.ndarray, 
                            forecast_times: List[int]) -> np.ndarray:
        """降尺度到1小时 - 均匀分配"""
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
                lat_var.long_name = 'latitude'
                
                lon_var[:] = self.lon_dst
                lon_var.units = 'degrees_east'
                lon_var.long_name = 'longitude'
                
                # 设置时间
                time_var.units = 'hours since 1970-01-01 00:00:00'
                time_var.calendar = 'gregorian'
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
                nc.title = 'ECMWF Hourly Precipitation'
                nc.source = 'ECMWF Forecast'
                nc.history = f'Created {datetime.now()}'
                nc.forecast_start = base_time_utc.strftime("%Y-%m-%d %H:%M UTC")
            
            os.rename(temp_path, output_path)
            return True
            
        except Exception as e:
            print(f"写入失败: {str(e)}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ECMWF降水处理')
    parser.add_argument('--file', type=str, required=True, help='输入文件')
    parser.add_argument('--output', type=str, help='输出文件路径（可选）')
    parser.add_argument('--output-dir', type=str, help='输出目录（可选，与--output互斥）')
    parser.add_argument('--test', action='store_true', help='测试模式')
    
    args = parser.parse_args()
    
    processor = PrecipitationProcessor()
    
    if args.test:
        # 仅分析
        info = processor.analyze_precipitation_structure(args.file)
        if info:
            print(f"\n预报时效: {info['forecast_times']}")
            print(f"平均降水值 (m): {[f'{v:.6f}' for v in info['values_avg']]}")
    else:
        # 确定输出路径
        output_path = args.output
        
        # 如果指定了输出目录但没有指定输出文件
        if args.output_dir and not args.output:
            # 获取基准时间
            base_time = Config.parse_time_from_filename(args.file)
            if base_time:
                bjt_time_str = Config.utc_to_bjt_str(base_time)
                output_filename = f"precip_0p01_1h_BJT_{bjt_time_str}.nc"
                output_path = os.path.join(args.output_dir, output_filename)
            else:
                print("无法解析基准时间，无法自动生成输出文件名")
                return
        
        success, output_file = processor.process_precipitation(
            input_file=args.file,
            output_file=output_path,
            skip_existing=False
        )
        
        if success:
            print(f"\n✓ 成功输出: {output_file}")
        else:
            print(f"\n✗ 失败")


if __name__ == "__main__":
    main()


'''
python EC_V2_PRE.py --file input.grib --output-dir ./output
'''