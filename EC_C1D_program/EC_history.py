#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECMWF 10米U/V风场数据批量处理 - 本地缓存优化版（修正文件名时间）
1. 从网络存储拷贝文件到本地
2. 在本地进行处理
3. 输出到指定目录（文件名使用北京时）

# 批量处理
python EC_history.py \
  --remote-dir /mnt/CMADAAS/DATA/NAFP/ECMF/C1D \
  --start-date 20251028 \
  --end-date 20260107 \
  --hours 00 12

# 保留本地缓存（用于调试）
python EC_history.py \
  --remote-dir /mnt/CMADAAS/DATA/NAFP/ECMF/C1D \
  --start-date 20260101 \
  --no-clean-cache

"""


#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
ECMWF 10米U/V风场数据批量处理 - 本地缓存优化版（修正文件名时间+纬度处理）
1. 从网络存储拷贝文件到本地
2. 在本地进行处理
3. 输出到指定目录（文件名使用北京时）

# 批量处理
python EC_history.py \
  --remote-dir /mnt/CMADAAS/DATA/NAFP/ECMF/C1D \
  --start-date 20251028 \
  --end-date 20260107 \
  --hours 00 12

# 保留本地缓存（用于调试）
python EC_history.py \
  --remote-dir /mnt/CMADAAS/DATA/NAFP/ECMF/C1D \
  --start-date 20260101 \
  --no-clean-cache

"""

import os
import sys
import shutil
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
    
    # 本地缓存目录
    LOCAL_CACHE_DIR = "/home/youqi/FWZX_forecast_DATA/history_EC/original_data"
    
    # 输出目录
    OUTPUT_DIR = "/home/youqi/FWZX_forecast_DATA/history_EC/interp_data"
    
    # 是否清理本地缓存
    CLEAN_CACHE = True
    
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


# ================= 日志配置 =================
def setup_logger(log_level=logging.INFO):
    """配置日志系统"""
    logger = logging.getLogger('ECMWF_Local_Processor')
    logger.setLevel(log_level)
    
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger


# ================= 文件管理类 =================
class FileManager:
    """管理文件的拷贝和清理"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger()
        self.copied_files = []  # 记录拷贝的文件
        
    def copy_to_local(self, src_u_file: str, src_v_file: str) -> Tuple[Optional[str], Optional[str]]:
        """将文件从网络存储拷贝到本地"""
        local_u_file = None
        local_v_file = None
        
        try:
            # 确保本地缓存目录存在
            os.makedirs(Config.LOCAL_CACHE_DIR, exist_ok=True)
            
            # 生成本地文件名（保留原始文件名）
            u_basename = os.path.basename(src_u_file)
            v_basename = os.path.basename(src_v_file)
            
            local_u_file = os.path.join(Config.LOCAL_CACHE_DIR, u_basename)
            local_v_file = os.path.join(Config.LOCAL_CACHE_DIR, v_basename)
            
            # 检查本地是否已存在（避免重复拷贝）
            copy_needed = False
            if not os.path.exists(local_u_file) or not os.path.exists(local_v_file):
                copy_needed = True
            else:
                # 检查文件大小是否一致
                src_u_size = os.path.getsize(src_u_file) if os.path.exists(src_u_file) else 0
                src_v_size = os.path.getsize(src_v_file) if os.path.exists(src_v_file) else 0
                local_u_size = os.path.getsize(local_u_file)
                local_v_size = os.path.getsize(local_v_file)
                
                if src_u_size != local_u_size or src_v_size != local_v_size:
                    copy_needed = True
            
            if copy_needed:
                self.logger.info(f"拷贝文件到本地...")
                start_time = time.time()
                
                if os.path.exists(src_u_file):
                    shutil.copy2(src_u_file, local_u_file)
                    self.copied_files.append(local_u_file)
                    self.logger.info(f"  已拷贝: {u_basename}")
                
                if os.path.exists(src_v_file):
                    shutil.copy2(src_v_file, local_v_file)
                    self.copied_files.append(local_v_file)
                    self.logger.info(f"  已拷贝: {v_basename}")
                
                elapsed = time.time() - start_time
                self.logger.info(f"拷贝完成，耗时: {elapsed:.1f}秒")
            else:
                self.logger.info(f"文件已存在于本地缓存，跳过拷贝")
            
            return local_u_file, local_v_file
            
        except Exception as e:
            self.logger.error(f"文件拷贝失败: {str(e)}")
            return None, None
    
    def cleanup_cache(self):
        """清理本地缓存文件"""
        if not Config.CLEAN_CACHE:
            return
        
        self.logger.info("清理本地缓存文件...")
        deleted_count = 0
        
        for file_path in self.copied_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_count += 1
            except Exception as e:
                self.logger.warning(f"无法删除文件 {file_path}: {str(e)}")
        
        self.logger.info(f"已清理 {deleted_count} 个缓存文件")
        self.copied_files = []


# ================= 核心处理类 =================
class ECMWFLocalProcessor:
    """ECMWF风场数据本地处理器"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger()
        self.file_manager = FileManager(logger)
        
        # 预计算目标网格（纬度升序）
        self.lat_dst, self.lon_dst = Config.get_target_grid()
        self.n_lat_dst, self.n_lon_dst = self.lat_dst.size, self.lon_dst.size
        
    def process_with_cache(self, remote_u_file: str, remote_v_file: str, 
                          base_time_utc: datetime) -> bool:
        """
        带本地缓存的处理流程
        1. 拷贝到本地
        2. 本地处理
        3. 输出结果（文件名使用北京时）
        
        Parameters
        ----------
        remote_u_file : str
            远程U文件路径
        remote_v_file : str
            远程V文件路径
        base_time_utc : datetime
            UTC基准时间（预报起始时间）
            
        Returns
        -------
        bool
            处理是否成功
        """
        total_start = time.time()
        
        try:
            # 阶段1: 检查远程文件是否存在
            u_basename = os.path.basename(remote_u_file)
            self.logger.info(f"开始处理: {u_basename}")
            
            if not os.path.exists(remote_u_file) or not os.path.exists(remote_v_file):
                self.logger.error("远程文件不存在")
                return False
            
            # 生成北京时文件名
            bjt_time_str = Config.utc_to_bjt_str(base_time_utc)
            output_filename = f"wspd_wdir_10m_0p01_1h_BJT_{bjt_time_str}.nc"
            
            # 检查输出是否已存在
            output_path = os.path.join(Config.OUTPUT_DIR, output_filename)
            if os.path.exists(output_path):
                self.logger.info(f"输出文件已存在: {output_filename}")
                return True
            
            # 阶段2: 拷贝到本地
            copy_start = time.time()
            local_u_file, local_v_file = self.file_manager.copy_to_local(
                remote_u_file, remote_v_file
            )
            
            if local_u_file is None or local_v_file is None:
                return False
            
            copy_time = time.time() - copy_start
            self.logger.info(f"文件拷贝耗时: {copy_time:.1f}秒")
            
            # 阶段3: 本地处理
            process_start = time.time()
            success = self._process_local(local_u_file, local_v_file, 
                                         base_time_utc, output_path)
            process_time = time.time() - process_start
            
            if success:
                total_time = time.time() - total_start
                self.logger.info(f"处理完成，总耗时: {total_time:.1f}秒 (拷贝: {copy_time:.1f}秒, 处理: {process_time:.1f}秒)")
                self.logger.info(f"输出文件: {output_filename}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"处理失败: {str(e)}")
            return False
    
    def _process_local(self, local_u_file: str, local_v_file: str, 
                      base_time_utc: datetime, output_path: str) -> bool:
        """在本地处理文件"""
        try:
            self.logger.info("开始本地处理...")
            
            # 1. 读取数据
            start_time = time.time()
            with pygrib.open(local_u_file) as grbs_u, pygrib.open(local_v_file) as grbs_v:
                # 获取所有时次
                steps_u = sorted({msg.step for msg in grbs_u})
                steps_v = sorted({msg.step for msg in grbs_v})
                common_steps = sorted(set(steps_u) & set(steps_v))
                
                if not common_steps:
                    self.logger.error("没有共同的预报时次")
                    return False
                
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
                
                self.logger.info(f"目标纬度: {self.lat_dst[0]:.2f} 到 {self.lat_dst[-1]:.2f} ({'升序' if self.lat_dst[-1] > self.lat_dst[0] else '降序'})")
                self.logger.info(f"源纬度: {lat_src[0]:.2f} 到 {lat_src[-1]:.2f} ({'升序' if lat_src[-1] > lat_src[0] else '降序'})")
                
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
            self.logger.info(f"文件写入完成: {write_time:.1f}秒, 大小: {os.path.getsize(output_path)/1024/1024:.1f} MB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"本地处理失败: {str(e)}", exc_info=True)
            return False


# ================= 批量处理类 =================
class BatchLocalProcessor:
    """批量本地处理器"""
    
    def __init__(self, remote_base_dir: str, logger=None):
        self.remote_base_dir = remote_base_dir
        self.logger = logger or setup_logger()
        self.processor = ECMWFLocalProcessor(logger)
        self.stats = {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}
    
    def find_files_for_datetime(self, date_str: str, hour: str) -> Tuple[Optional[str], Optional[str], Optional[datetime]]:
        """
        查找指定日期时次的文件，并解析时间
        
        Returns
        -------
        tuple
            (u_file_path, v_file_path, base_time_utc)
        """
        year = date_str[:4]
        remote_dir = os.path.join(self.remote_base_dir, year, date_str)
        
        u_file = os.path.join(remote_dir, f"ECMFC1D_10U_1_{date_str}{hour}_GLB_1.grib1")
        v_file = os.path.join(remote_dir, f"ECMFC1D_10V_1_{date_str}{hour}_GLB_1.grib1")
        
        # 解析基准时间（UTC）
        try:
            # 从文件名解析时间: ECMFC1D_10U_1_2026010100_GLB_1.grib1
            # 时间部分: 2026010100
            time_str = f"{date_str}{hour}"
            base_time_utc = datetime.strptime(time_str, "%Y%m%d%H")
        except Exception as e:
            self.logger.error(f"解析时间失败 {date_str}{hour}: {str(e)}")
            return None, None, None
        
        return u_file, v_file, base_time_utc
    
    def process_date_range(self, start_date: str, end_date: str, hours: List[str]):
        """处理日期范围"""
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        
        # 生成日期列表
        dates = []
        current_dt = start_dt
        while current_dt <= end_dt:
            dates.append(current_dt.strftime("%Y%m%d"))
            current_dt += timedelta(days=1)
        
        total_tasks = len(dates) * len(hours)
        self.logger.info(f"开始处理 {total_tasks} 个任务")
        self.logger.info(f"日期范围: {start_date} 到 {end_date}")
        self.logger.info(f"预报时次: {hours}")
        self.logger.info(f"本地缓存目录: {Config.LOCAL_CACHE_DIR}")
        self.logger.info(f"输出目录: {Config.OUTPUT_DIR}")
        
        task_start_time = time.time()
        
        for date_str in dates:
            for hour in hours:
                self.stats['total'] += 1
                task_id = f"{date_str}_{hour}"
                
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"处理任务 {self.stats['total']}/{total_tasks}: {task_id}")
                self.logger.info(f"{'='*60}")
                
                # 查找文件和解析时间
                remote_u_file, remote_v_file, base_time_utc = self.find_files_for_datetime(date_str, hour)
                
                if remote_u_file is None or remote_v_file is None or base_time_utc is None:
                    self.logger.warning(f"文件查找失败，跳过")
                    self.stats['skipped'] += 1
                    continue
                
                if not os.path.exists(remote_u_file) or not os.path.exists(remote_v_file):
                    self.logger.warning(f"文件不存在，跳过")
                    self.stats['skipped'] += 1
                    continue
                
                # 计算北京时用于显示信息
                bjt_time_str = Config.utc_to_bjt_str(base_time_utc)
                self.logger.info(f"UTC时间: {base_time_utc.strftime('%Y-%m-%d %H:%M:%S')}")
                self.logger.info(f"北京时: {bjt_time_str}")
                
                # 处理
                success = self.processor.process_with_cache(
                    remote_u_file, remote_v_file, base_time_utc
                )
                
                if success:
                    self.stats['success'] += 1
                else:
                    self.stats['failed'] += 1
                
                # 任务间短暂休息，避免资源竞争
                time.sleep(0.5)
        
        # 打印统计
        self._print_statistics(task_start_time)
        
        # 清理缓存
        self.processor.file_manager.cleanup_cache()
    
    def _print_statistics(self, start_time: float):
        """打印统计信息"""
        elapsed = time.time() - start_time
        
        summary = f"""
{'='*80}
批量处理完成！
总任务数: {self.stats['total']}
  成功: {self.stats['success']}
  失败: {self.stats['failed']}
  跳过: {self.stats['skipped']}
  成功率: {self.stats['success']/self.stats['total']*100:.1f}% ({self.stats['total']>0})
总耗时: {elapsed:.1f}秒
平均时间: {elapsed/self.stats['total']:.1f}秒/任务 ({self.stats['total']>0})

输出文件命名规则:
  wspd_wdir_10m_0p01_1h_BJT_YYYYMMDDHH.nc
  其中 YYYYMMDDHH 为北京时 (UTC+8)
  纬度顺序: 从南到北升序
{'='*80}
        """
        self.logger.info(summary)


# ================= 主函数 =================
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ECMWF风场数据批量处理（本地缓存版）')
    
    parser.add_argument('--remote-dir', type=str, required=True,
                       help='远程数据基础目录，如: /mnt/CMADAAS/DATA/NAFP/ECMF/C1D')
    
    parser.add_argument('--start-date', type=str, required=True,
                       help='开始日期，格式: YYYYMMDD')
    
    parser.add_argument('--end-date', type=str, default=None,
                       help='结束日期，格式: YYYYMMDD，默认为开始日期')
    
    parser.add_argument('--hours', type=str, nargs='+', default=['00', '12'],
                       help='预报时次，如: 00 12')
    
    parser.add_argument('--no-clean-cache', action='store_true', default=False,
                       help='不清理本地缓存文件')
    
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='跳过已存在的输出文件（默认启用）')
    
    args = parser.parse_args()
    
    # 配置参数
    if args.no_clean_cache:
        Config.CLEAN_CACHE = False
    
    # 设置日志
    logger = setup_logger()
    
    logger.info(f"{'='*80}")
    logger.info("ECMWF风场数据批量处理（本地缓存优化版 - 纬度修正）")
    logger.info(f"远程目录: {args.remote_dir}")
    logger.info(f"本地缓存: {Config.LOCAL_CACHE_DIR}")
    logger.info(f"输出目录: {Config.OUTPUT_DIR}")
    logger.info(f"日期范围: {args.start_date} 到 {args.end_date or args.start_date}")
    logger.info(f"预报时次: {args.hours}")
    logger.info(f"清理缓存: {'是' if Config.CLEAN_CACHE else '否'}")
    logger.info(f"跳过已存在文件: {'是' if args.skip_existing else '否'}")
    logger.info(f"纬度顺序: 从南到北升序")
    logger.info(f"{'='*80}")
    
    try:
        # 创建处理器
        processor = BatchLocalProcessor(args.remote_dir, logger)
        
        # 执行处理
        processor.process_date_range(
            args.start_date,
            args.end_date or args.start_date,
            args.hours
        )
        
        return 0
        
    except Exception as e:
        logger.error(f"批量处理失败: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    # 抑制警告
    warnings.filterwarnings("ignore")
    
    # 运行
    sys.exit(main())