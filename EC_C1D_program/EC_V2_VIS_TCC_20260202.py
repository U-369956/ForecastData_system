#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECMWF VIS和TCC数据处理模块 - 基于简化版思路
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
            "units": {"wspd": "m s-1", "wdir": "degree"}
        },
        "GUST": {
            "description": "10m wind gust",
            "grib_codes": {"value": "10FG3"},
            "requires_uv": False,
            "output_vars": ["gust"],
            "units": {"gust": "m s-1"}
        },
        "TEM": {
            "description": "2m temperature",
            "grib_codes": {"value": "TEM"},
            "requires_uv": False,
            "output_vars": ["temp"],
            "units": {"temp": "°C"},
            "conversion": "K_to_C"
        },
        "PRS": {
            "description": "Surface pressure",
            "grib_codes": {"value": "PRS"},
            "requires_uv": False,
            "output_vars": ["prs"],
            "units": {"prs": "Pa"}
        },
        "DPT": {
            "description": "2m dew point temperature",
            "grib_codes": {"value": "DPT"},
            "requires_uv": False,
            "output_vars": ["dpt"],
            "units": {"dpt": "°C"},
            "conversion": "K_to_C"
        },
        "RH": {
            "description": "2m relative humidity",
            "grib_codes": {"temp": "TEM", "prs": "PRS", "dpt": "DPT"},
            "requires_uv": False,
            "requires_calc": True,
            "output_vars": ["rh"],
            "units": {"rh": "%"}
        },
        # 新增要素
        "VIS": {
            "description": "Visibility",
            "grib_codes": {"value": "VIS"},
            "requires_uv": False,
            "output_vars": ["vis"],
            "units": {"vis": "m"}
        },
        "TCC": {
            "description": "Total cloud cover",
            "grib_codes": {"value": "TCC"},
            "requires_uv": False,
            "output_vars": ["tcc"],
            "units": {"tcc": "fraction"}
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


# ================= 处理类 =================
class ECProcessorSimple:
    """ECMWF数据处理器 - 简化版（类似小代码的思路）"""
    
    def __init__(self):
        # 预计算目标网格
        self.lat_dst, self.lon_dst = Config.get_target_grid()
        self.n_lat_dst, self.n_lon_dst = self.lat_dst.size, self.lon_dst.size
    
    def process_element(self, element: str, input_file: str, output_file: str = None,
                       skip_existing: bool = True) -> Tuple[bool, str]:
        """
        处理单个要素数据（简化接口）
        
        Args:
            element: 要素名称（VIS, TCC, TEM等）
            input_file: 输入GRIB文件路径
            output_file: 输出NC文件路径（可选）
            skip_existing: 是否跳过已存在的文件
        """
        try:
            print(f"处理{Config.ELEMENTS[element]['description']}数据: {os.path.basename(input_file)}")
            
            # 获取基准时间
            base_time = Config.parse_time_from_filename(input_file)
            if base_time is None:
                print("无法解析基准时间")
                return False, None
            
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
                return True, output_file
            
            # 处理数据
            success = self._process_scalar_data(
                element, input_file, base_time, output_file
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
            import traceback
            traceback.print_exc()
            return False, None
    
    def _process_scalar_data(self, element: str, input_file: str,
                            base_time_utc: datetime, output_path: str) -> bool:
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
                    return False
                
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
                return False
            
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
                precip_data = data_1h.copy()
                nan_mask = np.isnan(precip_data)
                if np.any(nan_mask):
                    precip_data[nan_mask] = -9999.0
                
                data_var[:] = precip_data
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
            
            return True
            
        except Exception as e:
            print(f"处理标量数据失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


# ================= 命令行接口 =================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ECMWF VIS和TCC数据处理')
    parser.add_argument('--element', type=str, required=True, 
                       choices=['VIS', 'TCC', 'TEM', 'PRS', 'DPT', 'GUST'],
                       help='要素名称：VIS(能见度), TCC(总云量), TEM(气温)等')
    parser.add_argument('--file', type=str, required=True, help='输入GRIB文件路径')
    parser.add_argument('--output', type=str, help='输出NC文件路径（可选）')
    parser.add_argument('--output-dir', type=str, help='输出目录（可选，与--output互斥）')
    
    args = parser.parse_args()
    
    processor = ECProcessorSimple()
    
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
    
    success, output_file = processor.process_element(
        element=args.element,
        input_file=args.file,
        output_file=output_path,
        skip_existing=False
    )
    
    if success:
        print(f"\n✓ 成功输出: {output_file}")
    else:
        print(f"\n✗ 处理失败")


if __name__ == "__main__":
    main()


'''
使用示例：
# 处理能见度
python EC_V2_VIS_TCC_20260202.py --element VIS --file /home/youqi/FWZX_forecast_DATA/data_demo/ECMFC1D_VIS_1_2026013100_GLB_1.grib1 --output-dir /home/youqi/FWZX_forecast_DATA/output

# 处理总云量
python EC_V2_VIS_TCC_20260202.py --element TCC --file /home/youqi/FWZX_forecast_DATA/data_demo/ECMFC1D_TCC_1_2026013100_GLB_1.grib1 --output-dir /home/youqi/FWZX_forecast_DATA/output

# 处理气温（测试）
python EC_VIS_TCC.py --element TEM --file ECMFC1D_TEM_1_2026010100_GLB_1.grib1 --output-dir ./output

# 指定输出文件
python EC_VIS_TCC.py --element VIS --file input.grib --output ./output/custom_name.nc
'''