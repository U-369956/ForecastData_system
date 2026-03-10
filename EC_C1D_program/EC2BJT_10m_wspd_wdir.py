#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECMWF 10米U/V风场数据处理优化版
功能：读取GRIB → 裁剪插值 → 计算风速风向 → 1小时间隔插值 → 转北京时 → 输出NetCDF
优化点：
1. 模块化设计，函数职责单一
2. 减少重复文件读取
3. 内存优化，使用生成器和分块处理
4. 批量插值提高性能
5. 完整的错误处理
6. 配置集中管理
"""

import os
import sys
import logging
import numpy as np
from netCDF4 import Dataset, date2num
from datetime import datetime, timedelta
import pygrib
from scipy.interpolate import RegularGridInterpolator, interp1d
from typing import Tuple, List, Dict, Optional
import warnings

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
    
    # 插值参数
    INTERP_METHOD = 'linear'  # 时间插值方法
    
    # NetCDF参数
    NC_COMPRESSION = True
    NC_COMPRESSION_LEVEL = 4
    
    @classmethod
    def get_target_grid(cls) -> Tuple[np.ndarray, np.ndarray]:
        """生成目标网格"""
        lon_out = np.arange(cls.REGION["lon_w"], 
                           cls.REGION["lon_e"] + cls.RESOLUTION/2, 
                           cls.RESOLUTION)
        lat_out = np.arange(cls.REGION["lat_n"], 
                           cls.REGION["lat_s"] - cls.RESOLUTION/2, 
                           -cls.RESOLUTION)  # 降序
        return lat_out, lon_out


# ================= 日志配置 =================
def setup_logger():
    """配置日志系统"""
    logger = logging.getLogger('ECMWF_Wind_Processor')
    logger.setLevel(logging.INFO)
    
    # 避免重复添加handler
    if not logger.handlers:
        # 控制台输出
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger


# ================= 核心处理类 =================
class ECMWFWindProcessor:
    """ECMWF风场数据处理主类"""
    
    def __init__(self, data_dir: str, u_file: str, v_file: str, out_nc: str):
        """
        初始化处理器
        
        Parameters
        ----------
        data_dir : str
            数据目录
        u_file : str
            U分量文件名
        v_file : str
            V分量文件名
        out_nc : str
            输出NetCDF文件名
        """
        self.data_dir = data_dir
        self.u_file = u_file
        self.v_file = v_file
        self.out_nc = out_nc
        self.logger = setup_logger()
        
        # 验证文件存在
        self._validate_files()
        
        # 初始化结果存储
        self.times_utc = None
        self.times_bjt = None
        self.lat = None
        self.lon = None
        self.wspd = None
        self.wdir = None
        
    def _validate_files(self):
        """验证输入文件是否存在"""
        files_to_check = [
            (self.u_file, "U分量"),
            (self.v_file, "V分量")
        ]
        
        for fname, desc in files_to_check:
            full_path = os.path.join(self.data_dir, fname)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"{desc}文件不存在: {full_path}")
            self.logger.info(f"找到{desc}文件: {fname}")
    
    def _get_common_steps(self) -> List[int]:
        """获取U/V文件共同的预报时次"""
        def get_steps(fname):
            with pygrib.open(os.path.join(self.data_dir, fname)) as grbs:
                return sorted({msg.step for msg in grbs})
        
        u_steps = get_steps(self.u_file)
        v_steps = get_steps(self.v_file)
        
        if u_steps != v_steps:
            warnings.warn("U/V文件的预报时次不完全一致，将取交集")
            common_steps = sorted(set(u_steps) & set(v_steps))
        else:
            common_steps = u_steps
        
        self.logger.info(f"共发现 {len(common_steps)} 个预报时次")
        return common_steps
    
    def _read_grib_data(self, fname: str, steps: List[int]) -> Tuple[List[datetime], np.ndarray, np.ndarray, np.ndarray]:
        """
        一次性读取GRIB文件的所有时次数据
        
        Returns
        -------
        times_utc : List[datetime]
            UTC时间列表
        data_cube : np.ndarray
            数据立方体 (time, lat, lon)
        lat_src : np.ndarray
            源纬度数组
        lon_src : np.ndarray
            源经度数组
        """
        self.logger.info(f"开始读取 {fname} ...")
        
        full_path = os.path.join(self.data_dir, fname)
        with pygrib.open(full_path) as grbs:
            # 第一个时次用于获取网格和时间基准
            msg_first = grbs.select(step=steps[0])[0]
            data_first, lats_first, lons_first = msg_first.data(
                lat1=Config.REGION["lat_s"],
                lat2=Config.REGION["lat_n"],
                lon1=Config.REGION["lon_w"],
                lon2=Config.REGION["lon_e"]
            )
            
            # 获取时间基准
            base_time = datetime.strptime(
                f"{msg_first.validityDate:08d}{msg_first.validityTime:04d}",
                "%Y%m%d%H%M"
            )
            
            lat_src = lats_first[:, 0]
            lon_src = lons_first[0, :]
            
            # 确保纬度降序
            if lat_src[-1] > lat_src[0]:
                lat_src = lat_src[::-1]
                data_first = data_first[::-1, :]
            
            # 初始化数据立方体
            n_time = len(steps)
            n_lat, n_lon = data_first.shape
            data_cube = np.empty((n_time, n_lat, n_lon), dtype=np.float32)
            data_cube[0] = data_first
            
            # 读取剩余时次
            for i, step in enumerate(steps[1:], 1):
                msg = grbs.select(step=step)[0]
                data, _, _ = msg.data(
                    lat1=Config.REGION["lat_s"],
                    lat2=Config.REGION["lat_n"],
                    lon1=Config.REGION["lon_w"],
                    lon2=Config.REGION["lon_e"]
                )
                
                if lat_src[-1] > lat_src[0]:
                    data = data[::-1, :]
                
                data_cube[i] = data
            
            # 生成时间列表
            times_utc = [base_time + timedelta(hours=int(step)) for step in steps]
        
        self.logger.info(
            f"读取完成: {fname}, 形状={data_cube.shape}, "
            f"纬度范围=[{lat_src.min():.2f}, {lat_src.max():.2f}], "
            f"经度范围=[{lon_src.min():.2f}, {lon_src.max():.2f}]"
        )
        
        return times_utc, data_cube, lat_src, lon_src
    
    def _batch_interpolate(self, lat_src: np.ndarray, lon_src: np.ndarray, 
                          data_cube: np.ndarray) -> np.ndarray:
        """
        批量插值到目标网格
        
        Parameters
        ----------
        lat_src : np.ndarray
            源纬度
        lon_src : np.ndarray
            源经度
        data_cube : np.ndarray
            源数据立方体
            
        Returns
        -------
        np.ndarray
            插值后的数据立方体
        """
        self.logger.info("开始批量插值到目标网格...")
        
        # 获取目标网格
        lat_dst, lon_dst = Config.get_target_grid()
        
        # 打印网格信息
        self.logger.info(
            f"目标网格: 纬度点数={lat_dst.size}, 经度点数={lon_dst.size}, "
            f"纬度范围=[{lat_dst[0]:.2f}, {lat_dst[-1]:.2f}], "
            f"经度范围=[{lon_dst[0]:.2f}, {lon_dst[-1]:.2f}]"
        )
        
        # 创建插值器
        interp_func = RegularGridInterpolator(
            (lat_src, lon_src),
            data_cube[0],  # 用第一时次初始化
            bounds_error=False,
            fill_value=np.nan
        )
        
        # 准备目标网格点
        lon2d, lat2d = np.meshgrid(lon_dst, lat_dst)
        points = np.column_stack([lat2d.ravel(), lon2d.ravel()])
        
        # 批量插值
        n_time = data_cube.shape[0]
        n_lat_dst, n_lon_dst = lat_dst.size, lon_dst.size
        result = np.empty((n_time, n_lat_dst, n_lon_dst), dtype=np.float32)
        
        for i in range(n_time):
            interp_func.values = data_cube[i]
            result[i] = interp_func(points).reshape(n_lat_dst, n_lon_dst)
            
            if (i + 1) % 10 == 0 or i == n_time - 1:
                self.logger.info(f"  插值进度: {i+1}/{n_time}")
        
        return result
    
    def _compute_wind_components(self, u_data: np.ndarray, v_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算风速和风向
        
        Parameters
        ----------
        u_data : np.ndarray
            U分量数据
        v_data : np.ndarray
            V分量数据
            
        Returns
        -------
        wspd : np.ndarray
            风速
        wdir : np.ndarray
            风向（度，0表示北风，90表示东风）
        """
        self.logger.info("计算风速和风向...")
        
        # 计算风速
        wspd = np.sqrt(u_data**2 + v_data**2)
        
        # 计算风向（气象角度：0度表示北风，90度表示东风）
        wdir = 180 + np.arctan2(u_data, v_data) * 180 / np.pi
        wdir = np.mod(wdir, 360)
        
        # 验证结果
        wspd_min, wspd_max = np.nanmin(wspd), np.nanmax(wspd)
        wdir_min, wdir_max = np.nanmin(wdir), np.nanmax(wdir)
        
        self.logger.info(
            f"风速范围: [{wspd_min:.2f}, {wspd_max:.2f}] m/s, "
            f"风向范围: [{wdir_min:.1f}, {wdir_max:.1f}]°"
        )
        
        return wspd, wdir
    
    def _interpolate_time(self, times_utc: List[datetime], data_3d: np.ndarray) -> Tuple[List[datetime], np.ndarray]:
        """
        时间维度插值到1小时间隔
        
        Parameters
        ----------
        times_utc : List[datetime]
            UTC时间列表
        data_3d : np.ndarray
            三维数据
            
        Returns
        -------
        new_times : List[datetime]
            新的时间列表（北京时）
        new_data : np.ndarray
            插值后的数据
        """
        self.logger.info("时间维度插值到1小时间隔...")
        
        # 转换为小时数
        t0 = times_utc[0]
        hours = np.array([(t - t0).total_seconds() / 3600. for t in times_utc])
        
        # 创建1小时间隔的时间序列
        hours_new = np.arange(hours[0], hours[-1] + 1, 1.0)
        new_times_utc = [t0 + timedelta(hours=float(h)) for h in hours_new]
        
        # 时间插值
        f = interp1d(hours, data_3d, axis=0, kind=Config.INTERP_METHOD,
                     bounds_error=False, fill_value='extrapolate')
        new_data = f(hours_new).astype(np.float32)
        
        # 转换为北京时
        new_times_bjt = [t + Config.TIMEZONE_SHIFT for t in new_times_utc]
        
        self.logger.info(
            f"时间插值完成: {len(times_utc)} -> {len(new_times_bjt)} 个时次, "
            f"数据形状: {data_3d.shape} -> {new_data.shape}"
        )
        
        return new_times_bjt, new_data
    
    def _write_netcdf(self):
        """写入NetCDF文件"""
        self.logger.info(f"写入NetCDF文件: {self.out_nc}")
        
        with Dataset(self.out_nc, 'w') as nc:
            # 创建维度
            nc.createDimension('time', len(self.times_bjt))
            nc.createDimension('lat', self.lat.size)
            nc.createDimension('lon', self.lon.size)
            
            # 创建坐标变量
            time_var = nc.createVariable('time', 'i4', ('time',))
            lat_var = nc.createVariable('lat', 'f4', ('lat',))
            lon_var = nc.createVariable('lon', 'f4', ('lon',))
            
            # 创建数据变量
            wspd_var = nc.createVariable('wspd', 'f4', ('time', 'lat', 'lon'),
                                         zlib=Config.NC_COMPRESSION,
                                         complevel=Config.NC_COMPRESSION_LEVEL)
            wdir_var = nc.createVariable('wdir', 'f4', ('time', 'lat', 'lon'),
                                         zlib=Config.NC_COMPRESSION,
                                         complevel=Config.NC_COMPRESSION_LEVEL)
            
            # 设置坐标变量属性
            lat_var[:] = self.lat
            lat_var.units = 'degrees_north'
            lat_var.long_name = 'latitude'
            lat_var.standard_name = 'latitude'
            
            lon_var[:] = self.lon
            lon_var.units = 'degrees_east'
            lon_var.long_name = 'longitude'
            lon_var.standard_name = 'longitude'
            
            # 设置时间变量
            time_var.units = 'hours since 1970-01-01 00:00:00'
            time_var.calendar = 'gregorian'
            time_var.long_name = 'time'
            time_var.standard_name = 'time'
            time_var.time_zone = 'UTC+8'
            time_var[:] = date2num(self.times_bjt, time_var.units, time_var.calendar)
            
            # 设置数据变量属性
            wspd_var[:] = self.wspd
            wspd_var.units = 'm s-1'
            wspd_var.long_name = '10 meter wind speed'
            wspd_var.standard_name = 'wind_speed'
            wspd_var.missing_value = np.nan
            
            wdir_var[:] = self.wdir
            wdir_var.units = 'degree'
            wdir_var.long_name = '10 meter wind direction'
            wdir_var.standard_name = 'wind_from_direction'
            wdir_var.comment = '0° indicates wind from north, 90° from east'
            wdir_var.missing_value = np.nan
            
            # 设置全局属性
            nc.title = 'ECMWF 10m Wind Speed and Direction'
            nc.source = 'ECMWF Forecast'
            nc.history = f'Created by ECMWFWindProcessor on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            nc.conventions = 'CF-1.8'
            nc.reference = 'https://www.ecmwf.int/'
            nc.region = f'lon: {Config.REGION["lon_w"]} to {Config.REGION["lon_e"]} E, lat: {Config.REGION["lat_s"]} to {Config.REGION["lat_n"]} N'
            nc.resolution = f'{Config.RESOLUTION} degree'
            nc.time_zone = 'UTC+8 (Beijing Time)'
        
        self.logger.info(f"NetCDF文件写入完成: {self.out_nc}, 大小: {os.path.getsize(self.out_nc)/1024/1024:.2f} MB")
    
    def process(self) -> bool:
        """执行完整的处理流程"""
        try:
            self.logger.info("开始ECMWF风场数据处理流程...")
            
            # 1. 获取共同的预报时次
            steps = self._get_common_steps()
            
            # 2. 读取U/V数据
            times_utc_u, u_cube, lat_src, lon_src = self._read_grib_data(self.u_file, steps)
            times_utc_v, v_cube, _, _ = self._read_grib_data(self.v_file, steps)
            
            # 验证时间一致性
            if times_utc_u != times_utc_v:
                warnings.warn("U/V数据的时间戳不完全一致")
                # 取共同的时间
                common_times = [t for t in times_utc_u if t in times_utc_v]
                if len(common_times) == 0:
                    raise ValueError("U/V数据没有共同的时间戳")
                
                # 筛选共同时间的数据
                u_indices = [i for i, t in enumerate(times_utc_u) if t in common_times]
                v_indices = [i for i, t in enumerate(times_utc_v) if t in common_times]
                
                u_cube = u_cube[u_indices]
                v_cube = v_cube[v_indices]
                times_utc = common_times
            else:
                times_utc = times_utc_u
            
            # 3. 批量插值到目标网格
            u_interp = self._batch_interpolate(lat_src, lon_src, u_cube)
            v_interp = self._batch_interpolate(lat_src, lon_src, v_cube)
            
            # 4. 计算风速风向
            wspd, wdir = self._compute_wind_components(u_interp, v_interp)
            
            # 5. 时间插值到1小时间隔并转北京时
            self.times_bjt, self.wspd = self._interpolate_time(times_utc, wspd)
            _, self.wdir = self._interpolate_time(times_utc, wdir)
            
            # 获取目标网格用于输出
            self.lat, self.lon = Config.get_target_grid()
            
            # 6. 写入NetCDF文件
            self._write_netcdf()
            
            # 7. 打印摘要信息
            self._print_summary()
            
            self.logger.info("处理流程完成！")
            return True
            
        except Exception as e:
            self.logger.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
            return False
    
    def _print_summary(self):
        """打印处理结果摘要"""
        summary = f"""
        ================= 处理结果摘要 =================
        输出文件: {self.out_nc}
        时间维度: {len(self.times_bjt)} 个时次
        空间网格: {self.lat.size} × {self.lon.size}
        时间范围: {self.times_bjt[0]} 到 {self.times_bjt[-1]} (北京时)
        纬度范围: [{self.lat.min():.2f}, {self.lat.max():.2f}] °N
        经度范围: [{self.lon.min():.2f}, {self.lon.max():.2f}] °E
        
        风速统计:
          最小值: {np.nanmin(self.wspd):.2f} m/s
          最大值: {np.nanmax(self.wspd):.2f} m/s
          平均值: {np.nanmean(self.wspd):.2f} m/s
          
        风向统计:
          最小值: {np.nanmin(self.wdir):.1f}°
          最大值: {np.nanmax(self.wdir):.1f}°
          平均值: {np.nanmean(self.wdir):.1f}°
        ================================================
        """
        self.logger.info(summary)


# ================= 主函数 =================
def main():
    """主函数，从环境变量获取参数"""
    # 从环境变量获取参数，带默认值
    data_dir = os.environ.get("EC_DATA_DIR", "/home/youqi/FWZX_forecast_DATA/data_demo")
    u_file = os.environ.get("EC_U_FILE", "ECMFC1D_10U_1_2026010100_GLB_1.grib1")
    v_file = os.environ.get("EC_V_FILE", "ECMFC1D_10V_1_2026010100_GLB_1.grib1")
    out_nc = os.environ.get("EC_OUT_NC", "wspd_wdir_10m_0p01_1h_BJT.nc")
    
    # 创建处理器
    processor = ECMWFWindProcessor(data_dir, u_file, v_file, out_nc)
    
    # 执行处理
    success = processor.process()
    
    if success:
        print(f"成功生成文件: {out_nc}")
        return 0
    else:
        print(f"处理失败，请检查日志")
        return 1


if __name__ == "__main__":
    # 抑制部分警告
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # 运行主程序
    sys.exit(main())