#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECMWF GRIB1数据CDO格式转换工具
负责将GRIB1文件转换为NetCDF格式，并进行质量检查
"""

import os
import sys
import logging
import argparse
import subprocess
import time
import pygrib
from netCDF4 import Dataset
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple, List


class CDOConverter:
    """使用CDO将GRIB1格式文件转换为NetCDF格式"""

    @staticmethod
    def _fill_missing_time_steps(nc_file: str, logger: logging.Logger = None) -> Optional[str]:
        """
        补全NC文件中缺失的时间点，将缺失时间点的数据设为np.nan

        Parameters
        ----------
        nc_file : str
            NetCDF文件路径
        logger : logging.Logger, optional
            日志记录器

        Returns
        -------
        str or None
            处理信息字符串，如果没有需要补全的时间点则返回None
        """
        try:
            with Dataset(nc_file, 'r') as ds:
                # 获取变量名（排除坐标变量）
                exclude_vars = ['time', 'latitude', 'longitude', 'lat', 'lon', 'level',
                                'forecast_reference_time', 'forecast_period',
                                'surface_altitude', 'height_above_ground',
                                'atmosphere_hybrid_sigma_pressure_coordinate']
                data_vars = [v for v in ds.variables.keys() if v not in exclude_vars]

                if not data_vars:
                    if logger:
                        logger.warning(f"未找到数据变量，跳过时间点补全")
                    return None

                var_name = data_vars[0]
                time_var = ds.variables['time']
                var = ds.variables[var_name]

                # 获取时间值（转换为小时）
                time_values = time_var[:]
                if hasattr(time_var, 'units') and 'hours since' in time_var.units:
                    # 时间值已经是小时数
                    time_hours = time_values
                else:
                    # 尝试获取 forecast_period 或计算时间差
                    time_hours = time_values

                # 检查并去除重复时间点
                time_to_data = {}  # 时间 -> (索引, 数据)
                duplicate_count = 0

                for i, t in enumerate(time_hours):
                    t_int = int(t)
                    if t_int in time_to_data:
                        duplicate_count += 1
                    # 保留最后一个（或可以改为保留第一个）
                    time_to_data[t_int] = (i, var[i])

                if duplicate_count > 0 and logger:
                    logger.info(f"发现 {duplicate_count} 个重复时间点，已去除")

                # 获取去重后的时间点集合
                time_set = set(time_to_data.keys())
                time_set_sorted = sorted(time_set)
                min_time = min(time_set)
                max_time = max(time_set)

                # 确定时间间隔并生成完整的时间序列
                var_name_lower = var_name.lower()
                if var_name_lower == '10fg3':
                    # 10fg3固定每3小时一个点
                    expected_times = list(range(min_time, max_time + 1, 3))
                elif var_name_lower in ['mn2t6', 'mx2t6']:
                    # MN2T6（过去6小时最低2m气温）和 MX2T6（过去6小时最高2m气温）
                    # 这两个要素是6小时间隔，需要检查连续缺失并用NaN补全
                    # 生成完整的6小时间隔序列
                    expected_times = list(range(min_time, max_time + 1, 6))
                else:
                    # 其他变量：0-72小时每3小时，78小时以后每6小时
                    expected_times = []
                    # 0-72小时，每3小时一个点
                    for t in range(min_time, min(73, max_time + 1), 3):
                        expected_times.append(t)
                    # 78小时到最后，每6小时一个点
                    for t in range(78, max_time + 1, 6):
                        expected_times.append(t)

                # 计算缺失的时间点
                missing_times = [t for t in expected_times if t not in time_set]

                # 检查连续缺失
                if missing_times:
                    consecutive_missing = []
                    current_streak = []

                    for i in range(len(missing_times)):
                        if i == 0 or missing_times[i] != missing_times[i-1] + 6:  # MN2T6/MX2T6间隔为6小时
                            # 检查当前streak长度
                            if len(current_streak) > 0:
                                consecutive_missing.extend(current_streak)
                            current_streak = [missing_times[i]]
                        else:
                            current_streak.append(missing_times[i])

                    # 处理最后一个streak
                    if len(current_streak) > 0:
                        consecutive_missing.extend(current_streak)

                    # 检查是否有连续2个以上缺失
                    # 对于MN2T6/MX2T6，连续2个缺失意味着实际数据间隔是12小时
                    if len(consecutive_missing) >= 2:
                        # 有连续缺失，不生成
                        if logger:
                            logger.error(f"{var_name} 检测到连续缺失: {consecutive_missing}，不生成文件")
                        raise Exception(f"连续缺失时间点超过1个")

                # 获取空间维度大小（提前定义，供所有分支使用）
                var_dims = var.dimensions
                spatial_shape = var.shape[1:]
                nan_data = np.full(spatial_shape, np.nan, dtype=var.dtype if var.dtype.kind == 'f' else np.float32)

                if not missing_times and duplicate_count == 0:
                    if logger:
                        logger.debug(f"时间序列完整，无重复，无需处理")
                    return None

                # 如果只有去重，没有补全
                if not missing_times and duplicate_count > 0:
                    # 创建临时文件用于写入去重后的数据
                    import tempfile
                    temp_file = nc_file + '.filling'

                    # 构建完整数据
                    time_dim_size = len(expected_times)
                    new_data = np.full((time_dim_size,) + spatial_shape, np.nan, dtype=nan_data.dtype)

                    # 填充去重后的原始数据
                    time_to_index = {t: i for i, t in enumerate(expected_times)}
                    for t, (orig_idx, data) in time_to_data.items():
                        if t in time_to_index:
                            new_data[time_to_index[t]] = data

                    # 写入新的NC文件
                    with Dataset(temp_file, 'w', format='NETCDF4') as new_ds:
                        # 复制全局属性
                        for attr_name in ds.ncattrs():
                            new_ds.setncattr(attr_name, ds.getncattr(attr_name))

                        # 创建维度
                        for dim_name, dim in ds.dimensions.items():
                            if dim_name == 'time':
                                new_ds.createDimension(dim_name, len(expected_times))
                            else:
                                new_ds.createDimension(dim_name, len(dim))

                        # 创建时间变量
                        new_time_var = new_ds.createVariable('time', time_var.dtype, ('time',))
                        for attr_name in time_var.ncattrs():
                            new_time_var.setncattr(attr_name, time_var.getncattr(attr_name))
                        new_time_var[:] = expected_times

                        # 复制坐标变量
                        for coord_var in ['latitude', 'longitude', 'lat', 'lon', 'level',
                                         'forecast_reference_time', 'forecast_period',
                                         'surface_altitude', 'height_above_ground',
                                         'atmosphere_hybrid_sigma_pressure_coordinate']:
                            if coord_var in ds.variables:
                                orig_coord = ds.variables[coord_var]
                                new_coord = new_ds.createVariable(
                                    coord_var, orig_coord.dtype, orig_coord.dimensions
                                )
                                for attr_name in orig_coord.ncattrs():
                                    new_coord.setncattr(attr_name, orig_coord.getncattr(attr_name))
                                new_coord[:] = orig_coord[:]

                        # 创建数据变量并写入数据
                        new_var = new_ds.createVariable(
                            var_name, nan_data.dtype, var_dims,
                            fill_value=np.nan
                        )
                        for attr_name in var.ncattrs():
                            new_var.setncattr(attr_name, var.getncattr(attr_name))
                        new_var[:] = new_data

                    # 替换原文件
                    import shutil
                    shutil.move(temp_file, nc_file)

                    return f"去除 {duplicate_count} 个重复时间点"

                if logger:
                    logger.info(f"发现缺失时间点: {missing_times}，开始补全...")

                # 构建完整数据（spatial_shape 和 nan_data 已在前面定义）
                time_dim_size = len(expected_times)

                # 创建新的数据数组
                new_data = np.full((time_dim_size,) + spatial_shape, np.nan, dtype=nan_data.dtype)

                # 填充去重后的原始数据
                time_to_index = {t: i for i, t in enumerate(expected_times)}
                for t, (orig_idx, data) in time_to_data.items():
                    if t in time_to_index:
                        new_data[time_to_index[t]] = data

                # 创建临时文件
                import tempfile
                temp_file = nc_file + '.filling'

                # 写入新的NC文件
                with Dataset(temp_file, 'w', format='NETCDF4') as new_ds:
                    # 复制全局属性
                    for attr_name in ds.ncattrs():
                        new_ds.setncattr(attr_name, ds.getncattr(attr_name))

                    # 创建维度
                    for dim_name, dim in ds.dimensions.items():
                        if dim_name == 'time':
                            new_ds.createDimension(dim_name, len(expected_times))
                        else:
                            new_ds.createDimension(dim_name, len(dim))

                    # 创建时间变量
                    new_time_var = new_ds.createVariable('time', time_var.dtype, ('time',))
                    for attr_name in time_var.ncattrs():
                        new_time_var.setncattr(attr_name, time_var.getncattr(attr_name))
                    new_time_var[:] = expected_times

                    # 复制坐标变量
                    for coord_var in ['latitude', 'longitude', 'lat', 'lon', 'level',
                                     'forecast_reference_time', 'forecast_period',
                                     'surface_altitude', 'height_above_ground',
                                     'atmosphere_hybrid_sigma_pressure_coordinate']:
                        if coord_var in ds.variables:
                            orig_coord = ds.variables[coord_var]
                            new_coord = new_ds.createVariable(
                                coord_var, orig_coord.dtype, orig_coord.dimensions
                            )
                            for attr_name in orig_coord.ncattrs():
                                new_coord.setncattr(attr_name, orig_coord.getncattr(attr_name))
                            new_coord[:] = orig_coord[:]

                    # 创建数据变量并写入数据
                    new_var = new_ds.createVariable(
                        var_name, nan_data.dtype, var_dims,
                        fill_value=np.nan
                    )
                    for attr_name in var.ncattrs():
                        new_var.setncattr(attr_name, var.getncattr(attr_name))
                    new_var[:] = new_data

                # 替换原文件
                import shutil
                shutil.move(temp_file, nc_file)

                # 返回处理信息
                info_parts = []
                if duplicate_count > 0:
                    info_parts.append(f"去除 {duplicate_count} 个重复时间点")
                if missing_times:
                    info_parts.append(f"补全 {len(missing_times)} 个缺失时间点: {missing_times}")
                return ', '.join(info_parts)

        except Exception as e:
            raise Exception(f"补全时间序列失败: {str(e)}")

    @staticmethod
    def grib1_to_netcdf(grib1_file: str, output_nc_file: str,
                       logger: logging.Logger = None,
                       check_step_count: bool = True,
                       temp_dir: Optional[str] = None) -> Tuple[bool, Dict]:
        """
        使用CDO将GRIB1格式文件转换为NetCDF格式

        Parameters
        ----------
        grib1_file : str
            输入的GRIB1格式文件路径
        output_nc_file : str
            输出的NetCDF文件路径
        logger : logging.Logger, optional
            日志记录器
        check_step_count : bool
            是否检查时效数一致性
        temp_dir : str, optional
            临时文件夹路径，用于存储中间文件。如果为None，使用输出文件的父目录

        Returns
        -------
        tuple
            (是否成功, 信息字典)
        """
        info = {
            'grib_file': grib1_file,
            'output_nc': output_nc_file,
            'grib_steps': None,
            'nc_steps': None,
            'var_name': None,
            'var_long_name': None,
            'conversion_time': None,
            'file_size_mb': None
        }

        try:
            # 检查输入文件是否存在
            if not os.path.exists(grib1_file):
                error_msg = f"输入文件不存在: {grib1_file}"
                if logger:
                    logger.error(error_msg)
                return False, {**info, 'error': error_msg}

            # 步骤1: 读取原始GRIB1文件时效数
            if check_step_count:
                try:
                    with pygrib.open(grib1_file) as grbs:
                        grib_steps = sorted({msg.step for msg in grbs})
                        grib_step_count = len(grib_steps)
                        info['grib_steps'] = grib_steps
                        if logger:
                            logger.info(f"原始GRIB1时效数: {grib_step_count} 个，范围: {min(grib_steps)}-{max(grib_steps)}")
                except Exception as e:
                    error_msg = f"读取GRIB1文件时效数失败: {str(e)}"
                    if logger:
                        logger.error(error_msg)
                    return False, {**info, 'error': error_msg}

            # 步骤2: 设置临时文件路径
            if temp_dir is None:
                temp_dir = os.path.dirname(output_nc_file)
            else:
                os.makedirs(temp_dir, exist_ok=True)

            basename = os.path.basename(output_nc_file)
            temp_nc_file = os.path.join(temp_dir, basename + '.tmp')

            # 确保输出目录存在
            final_output_dir = os.path.dirname(output_nc_file)
            os.makedirs(final_output_dir, exist_ok=True)

            if logger:
                logger.info(f"使用临时目录: {temp_dir}")
                logger.debug(f"临时文件路径: {temp_nc_file}")

            # 步骤3: 执行CDO转换
            # -f nc: 输出格式为NetCDF
            # sorttaxis: 按时间轴排序，确保时间维度正确排列
            cdo_cmd = ['cdo', '-f', 'nc', 'sorttaxis', grib1_file, temp_nc_file]

            if logger:
                logger.info(f"开始CDO格式转换: {grib1_file} -> {temp_nc_file}")
                logger.debug(f"执行命令: {' '.join(cdo_cmd)}")

            # 执行CDO命令
            start_time = time.time()
            result = subprocess.run(
                cdo_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            conversion_time = time.time() - start_time
            info['conversion_time'] = conversion_time

            if logger:
                logger.info(f"CDO格式转换完成，耗时: {conversion_time:.1f}秒")
                logger.debug(f"CDO输出: {result.stdout}")
                if result.stderr:
                    logger.debug(f"CDO错误输出: {result.stderr}")

            # 检查临时文件是否存在
            if not os.path.exists(temp_nc_file):
                error_msg = f"CDO转换失败，输出文件不存在: {temp_nc_file}"
                if logger:
                    logger.error(error_msg)
                return False, {**info, 'error': error_msg}

            # 步骤4: 检查NC文件时效数
            if check_step_count:
                try:
                    with Dataset(temp_nc_file, 'r') as ds:
                        if 'time' in ds.variables:
                            time_var = ds.variables['time']
                            nc_step_count = len(time_var[:])
                            info['nc_steps'] = nc_step_count
                            if logger:
                                logger.info(f"转换后NC文件时效数: {nc_step_count} 个")

                            # 获取变量名（排除坐标变量）
                            exclude_vars = ['time', 'latitude', 'longitude', 'lat', 'lon', 'level',
                                            'forecast_reference_time', 'forecast_period',
                                            'surface_altitude', 'height_above_ground',
                                            'atmosphere_hybrid_sigma_pressure_coordinate']
                            data_vars = [v for v in ds.variables.keys() if v not in exclude_vars]
                            if data_vars:
                                var_name = data_vars[0]
                                info['var_name'] = var_name
                                var = ds.variables[var_name]
                                info['var_long_name'] = getattr(var, 'long_name', 'N/A')
                                if logger:
                                    logger.info(f"数据变量: {var_name} ({info['var_long_name']})")

                except Exception as e:
                    error_msg = f"读取NC文件时效数失败: {str(e)}"
                    if logger:
                        logger.error(error_msg)
                    return False, {**info, 'error': error_msg}

                # 步骤4: 对比时效数（仅供参考，不作为失败条件）
                # CDO会自动处理重复时间点，所以GRIB1和NC的时间点数量可能不一致
                if info['grib_steps'] is not None and info['nc_steps'] is not None:
                    grib_count = len(info['grib_steps'])
                    nc_count = info['nc_steps']

                    if grib_count != nc_count:
                        # 只记录警告，不删除文件，继续后续检查
                        warning_msg = (f"时效数不一致（可能是重复时间点被CDO处理）: "
                                     f"GRIB1: {grib_count} 个, NC: {nc_count} 个, "
                                     f"差值: {abs(grib_count - nc_count)}")
                        if logger:
                            logger.warning(warning_msg)
                    else:
                        if logger:
                            logger.info(f"✓ 时效数一致: {grib_count} 个")

                # 步骤5: 检查时间点数量及时间维度是否符合规范
                try:
                    with Dataset(temp_nc_file, 'r') as ds:
                        if 'time' in ds.variables:
                            time_var = ds.variables['time']
                            time_count = len(time_var[:])
                            var_name = info.get('var_name', '')

                            # 记录时间点数量（不进行硬性检查，只做信息记录）
                            if logger:
                                logger.info(f"NC文件时间点数量: {time_count} 个")

                            # 获取时间值（转换为小时）
                            time_values = time_var[:]
                            if hasattr(time_var, 'units') and 'hours since' in time_var.units:
                                # 时间值已经是小时数
                                time_hours = time_values
                            else:
                                # 尝试获取 forecast_period 或计算时间差
                                time_hours = time_values

                            # 排序时间值
                            time_hours_sorted = sorted(time_hours)
                            if logger:
                                logger.debug(f"时间点列表: {list(time_hours_sorted)}")

                            # 步骤6: 检查第一个时间点
                            first_time = min(time_hours)
                            var_name_lower = var_name.lower()

                            # 根据不同要素检查第一个时间点
                            if var_name_lower == '10fg3':
                                expected_time = 3
                                if first_time != expected_time:
                                    error_msg = f"10fg3第一个时间点必须为{expected_time}，当前为 {first_time}"
                                    if logger:
                                        logger.error(error_msg)

                                    if os.path.exists(temp_nc_file):
                                        os.remove(temp_nc_file)
                                    return False, {**info, 'error': error_msg}
                                else:
                                    if logger:
                                        logger.info(f"✓ 第一个时间点正确: {first_time} 小时")
                            elif var_name_lower in ['mn2t6', 'mx2t6']:
                                # MN2T6（过去6小时最低2m气温）和 MX2T6（过去6小时最高2m气温）
                                # 第一个时间点应该是6小时
                                expected_time = 6
                                if first_time != expected_time:
                                    error_msg = f"{var_name}第一个时间点必须为{expected_time}，当前为 {first_time}"
                                    if logger:
                                        logger.error(error_msg)

                                    if os.path.exists(temp_nc_file):
                                        os.remove(temp_nc_file)
                                    return False, {**info, 'error': error_msg}
                                else:
                                    if logger:
                                        logger.info(f"✓ 第一个时间点正确: {first_time} 小时")
                            else:
                                # 其他要素第一个时间点必须为0
                                expected_time = 0
                                if first_time != expected_time:
                                    error_msg = f"{var_name} 第一个时间点必须为{expected_time}，当前为 {first_time}"
                                    if logger:
                                        logger.error(error_msg)

                                    if os.path.exists(temp_nc_file):
                                        os.remove(temp_nc_file)
                                    return False, {**info, 'error': error_msg}
                                else:
                                    if logger:
                                        logger.info(f"✓ 第一个时间点正确: {first_time} 小时")

                            # 步骤7: 检查最大时间点（非10fg3变量）
                            if var_name.lower() != '10fg3':
                                max_time = max(time_hours)
                                if max_time < 25:
                                    error_msg = (f"{var_name} 最大时间点必须 >= 25，当前为 {max_time}")
                                    if logger:
                                        logger.error(error_msg)

                                    if os.path.exists(temp_nc_file):
                                        os.remove(temp_nc_file)
                                    return False, {**info, 'error': error_msg}
                                else:
                                    if logger:
                                        logger.info(f"✓ 最大时间点符合要求: {max_time} 小时")

                            # 步骤8: 检查连续缺失的时间点
                            # 将时间点转换为整数并去重
                            time_set = set(int(t) for t in time_hours)
                            time_set_sorted = sorted(time_set)

                            # 计算缺失的时间点
                            min_time = min(time_set)
                            max_time = max(time_set)

                            # 确定预期时间序列并检查连续缺失
                            var_name_lower = var_name.lower()
                            if var_name_lower == '10fg3':
                                # 10fg3固定每3小时一个点
                                expected_times = list(range(min_time, max_time + 1, 3))
                                time_interval = 3
                            elif var_name_lower in ['mn2t6', 'mx2t6']:
                                # MN2T6和MX2T6本来就是6小时间隔，不需要补全
                                expected_times = sorted(list(time_set))
                                time_interval = 6
                            else:
                                # 其他变量：0-72小时每3小时，78小时以后每6小时
                                expected_times = []
                                # 0-72小时，每3小时一个点
                                for t in range(min_time, min(73, max_time + 1), 3):
                                    expected_times.append(t)
                                # 78小时到最后，每6小时一个点
                                for t in range(78, max_time + 1, 6):
                                    expected_times.append(t)

                                # 确定连续缺失的判断间隔
                                # 对于不同段使用不同的间隔
                                time_intervals = {}  # {时间点: 间隔}
                                for t in expected_times:
                                    if t <= 72:
                                        time_intervals[t] = 3
                                    else:
                                        time_intervals[t] = 6

                            missing_times = [t for t in expected_times if t not in time_set]

                            # 检查连续缺失
                            if missing_times:
                                consecutive_missing = []
                                current_streak = []

                                for i in range(len(missing_times)):
                                    # 判断是否连续
                                    if var_name.lower() == '10fg3':
                                        # 10fg3：固定间隔3
                                        is_continuous = (i > 0 and missing_times[i] == missing_times[i-1] + time_interval)
                                    else:
                                        # 其他变量：使用动态间隔
                                        if i > 0:
                                            current_time = missing_times[i]
                                            prev_time = missing_times[i-1]
                                            # 根据前一个时间点确定间隔
                                            interval = 3 if prev_time <= 72 else 6
                                            is_continuous = (current_time == prev_time + interval)
                                        else:
                                            is_continuous = False

                                    if not is_continuous:
                                        # 开始新的连续序列
                                        if current_streak:
                                            consecutive_missing.append(current_streak)
                                        current_streak = [missing_times[i]]
                                    else:
                                        # 继续当前序列
                                        current_streak.append(missing_times[i])
                                if current_streak:
                                    consecutive_missing.append(current_streak)

                                # 找出连续缺失>=2的序列
                                long_streaks = [s for s in consecutive_missing if len(s) >= 2]

                                if long_streaks:
                                    streak_str = ', '.join([f"[{min(s)}-{max(s)}]" for s in long_streaks])
                                    error_msg = (f"发现连续缺失2个或以上时间点: {streak_str}")
                                    if logger:
                                        logger.error(error_msg)

                                    if os.path.exists(temp_nc_file):
                                        os.remove(temp_nc_file)
                                    return False, {**info, 'error': error_msg}
                                else:
                                    if missing_times:
                                        if logger:
                                            logger.info(f"✓ 无连续2个以上时间点缺失，单独缺失: {missing_times}")
                                    else:
                                        if logger:
                                            logger.info(f"✓ 时间点完整无缺失")
                            else:
                                if logger:
                                    logger.info(f"✓ 时间点完整无缺失")

                except Exception as e:
                    error_msg = f"检查时间点失败: {str(e)}"
                    if logger:
                        logger.error(error_msg)
                    return False, {**info, 'error': error_msg}
            else:
                # 如果不检查时效数，只进行时间点检查
                try:
                    with Dataset(temp_nc_file, 'r') as ds:
                        # 获取变量名（排除坐标变量）
                        exclude_vars = ['time', 'latitude', 'longitude', 'lat', 'lon', 'level',
                                        'forecast_reference_time', 'forecast_period',
                                        'surface_altitude', 'height_above_ground',
                                        'atmosphere_hybrid_sigma_pressure_coordinate']
                        data_vars = [v for v in ds.variables.keys() if v not in exclude_vars]
                        if data_vars:
                            var_name = data_vars[0]
                            info['var_name'] = var_name
                            var = ds.variables[var_name]
                            info['var_long_name'] = getattr(var, 'long_name', 'N/A')
                            if logger:
                                logger.info(f"数据变量: {var_name} ({info['var_long_name']})")

                        # 检查时间点数量及质量
                        if 'time' in ds.variables:
                            time_var = ds.variables['time']
                            time_count = len(time_var[:])

                            # 记录时间点数量（不进行硬性检查，只做信息记录）
                            if logger:
                                logger.info(f"NC文件时间点数量: {time_count} 个")

                            # 获取时间值（转换为小时）
                            time_values = time_var[:]
                            if hasattr(time_var, 'units') and 'hours since' in time_var.units:
                                # 时间值已经是小时数
                                time_hours = time_values
                            else:
                                # 尝试获取 forecast_period 或计算时间差
                                time_hours = time_values

                            # 排序时间值
                            time_hours_sorted = sorted(time_hours)
                            if logger:
                                logger.debug(f"时间点列表: {list(time_hours_sorted)}")

                            # 检查第一个时间点
                            first_time = min(time_hours)
                            var_name_lower = var_name.lower()

                            # 根据不同要素检查第一个时间点
                            if var_name_lower == '10fg3':
                                expected_time = 3
                                if first_time != expected_time:
                                    error_msg = f"10fg3第一个时间点必须为{expected_time}，当前为 {first_time}"
                                    if logger:
                                        logger.error(error_msg)

                                    if os.path.exists(temp_nc_file):
                                        os.remove(temp_nc_file)
                                    return False, {**info, 'error': error_msg}
                                else:
                                    if logger:
                                        logger.info(f"✓ 第一个时间点正确: {first_time} 小时")
                            elif var_name_lower in ['mn2t6', 'mx2t6']:
                                # MN2T6（过去6小时最低2m气温）和 MX2T6（过去6小时最高2m气温）
                                # 第一个时间点应该是6小时
                                expected_time = 6
                                if first_time != expected_time:
                                    error_msg = f"{var_name}第一个时间点必须为{expected_time}，当前为 {first_time}"
                                    if logger:
                                        logger.error(error_msg)

                                    if os.path.exists(temp_nc_file):
                                        os.remove(temp_nc_file)
                                    return False, {**info, 'error': error_msg}
                                else:
                                    if logger:
                                        logger.info(f"✓ 第一个时间点正确: {first_time} 小时")
                            else:
                                # 其他要素第一个时间点必须为0
                                expected_time = 0
                                if first_time != expected_time:
                                    error_msg = f"{var_name} 第一个时间点必须为{expected_time}，当前为 {first_time}"
                                    if logger:
                                        logger.error(error_msg)

                                    if os.path.exists(temp_nc_file):
                                        os.remove(temp_nc_file)
                                    return False, {**info, 'error': error_msg}
                                else:
                                    if logger:
                                        logger.info(f"✓ 第一个时间点正确: {first_time} 小时")

                            # 检查最大时间点（非10fg3变量）
                            if var_name.lower() != '10fg3':
                                max_time = max(time_hours)
                                if max_time < 25:
                                    error_msg = (f"{var_name} 最大时间点必须 >= 25，当前为 {max_time}")
                                    if logger:
                                        logger.error(error_msg)

                                    if os.path.exists(temp_nc_file):
                                        os.remove(temp_nc_file)
                                    return False, {**info, 'error': error_msg}
                                else:
                                    if logger:
                                        logger.info(f"✓ 最大时间点符合要求: {max_time} 小时")

                            # 检查连续缺失的时间点
                            time_set = set(int(t) for t in time_hours)
                            min_time = min(time_set)
                            max_time = max(time_set)

                            var_name_lower = var_name.lower()
                            if var_name_lower == '10fg3':
                                # 10fg3检查整个序列
                                expected_times = list(range(min_time, max_time + 1, 3))  # 每3小时一个点
                            elif var_name_lower in ['mn2t6', 'mx2t6']:
                                # MN2T6和MX2T6本来就是6小时间隔，不需要补全检查
                                expected_times = sorted(list(time_set))
                            else:
                                # 其他变量只检查到25小时
                                check_max = min(max_time, 25)
                                expected_times = list(range(min_time, check_max + 1))

                            missing_times = [t for t in expected_times if t not in time_set]

                            # 检查连续缺失
                            if missing_times:
                                consecutive_missing = []
                                current_streak = []

                                for i in range(len(missing_times)):
                                    if i == 0 or missing_times[i] != missing_times[i-1] + 1:
                                        if current_streak:
                                            consecutive_missing.append(current_streak)
                                        current_streak = [missing_times[i]]
                                    else:
                                        current_streak.append(missing_times[i])
                                if current_streak:
                                    consecutive_missing.append(current_streak)

                                # 找出连续缺失>=2的序列
                                long_streaks = [s for s in consecutive_missing if len(s) >= 2]

                                if long_streaks:
                                    streak_str = ', '.join([f"[{min(s)}-{max(s)}]" for s in long_streaks])
                                    error_msg = (f"发现连续缺失2个或以上时间点: {streak_str}")
                                    if logger:
                                        logger.error(error_msg)

                                    if os.path.exists(temp_nc_file):
                                        os.remove(temp_nc_file)
                                    return False, {**info, 'error': error_msg}
                                else:
                                    if missing_times:
                                        if logger:
                                            logger.info(f"✓ 无连续2个以上时间点缺失，单独缺失: {missing_times}")
                                    else:
                                        if logger:
                                            logger.info(f"✓ 时间点完整无缺失")
                            else:
                                if logger:
                                    logger.info(f"✓ 时间点完整无缺失")

                except Exception as e:
                    if logger:
                        logger.warning(f"读取NC文件信息失败: {str(e)}")

            # 检查文件大小
            file_size = os.path.getsize(temp_nc_file) / 1024 / 1024  # MB
            info['file_size_mb'] = file_size

            # 步骤9: 补全缺失的时间点，确保时间序列完整
            try:
                filled_info = CDOConverter._fill_missing_time_steps(
                    temp_nc_file, logger
                )
                if filled_info:
                    if logger:
                        logger.info(f"✓ 时间序列补全完成: {filled_info}")
            except Exception as e:
                error_msg = f"补全时间序列失败: {str(e)}"
                if logger:
                    logger.error(error_msg)
                # 清理临时文件
                if os.path.exists(temp_nc_file):
                    os.remove(temp_nc_file)
                return False, {**info, 'error': error_msg}

            # 所有检查通过，将临时文件移动到最终位置
            import shutil
            if logger:
                logger.info(f"所有检查通过，移动文件: {temp_nc_file} -> {output_nc_file}")

            shutil.move(temp_nc_file, output_nc_file)

            if logger:
                logger.info(f"NetCDF文件创建成功: {output_nc_file} ({file_size:.1f} MB)")

            return True, info

        except subprocess.CalledProcessError as e:
            error_msg = f"CDO命令执行失败: {e.stderr if e.stderr else str(e)}"
            if logger:
                logger.error(error_msg)
            # 清理临时文件
            if 'temp_nc_file' in locals() and os.path.exists(temp_nc_file):
                os.remove(temp_nc_file)
                if logger:
                    logger.info(f"已清理临时文件: {temp_nc_file}")
            return False, {**info, 'error': error_msg}
        except FileNotFoundError:
            error_msg = "未找到cdo命令，请确保CDO已安装并在PATH中"
            if logger:
                logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"CDO格式转换失败: {str(e)}"
            if logger:
                logger.error(error_msg)
            # 清理临时文件
            if 'temp_nc_file' in locals() and os.path.exists(temp_nc_file):
                os.remove(temp_nc_file)
                if logger:
                    logger.info(f"已清理临时文件: {temp_nc_file}")
            return False, {**info, 'error': error_msg}

    @staticmethod
    def convert_multiple_files(grib1_files: List[str], output_dir: str,
                              logger: logging.Logger = None,
                              check_step_count: bool = True,
                              temp_dir: Optional[str] = None,
                              skip_existing: bool = False) -> Dict[str, Dict]:
        """
        批量转换多个GRIB1文件为NetCDF格式

        Parameters
        ----------
        grib1_files : list of str
            输入的GRIB1格式文件路径列表
        output_dir : str
            输出目录
        logger : logging.Logger, optional
            日志记录器
        check_step_count : bool
            是否检查时效数一致性
        temp_dir : str, optional
            临时文件夹路径，用于存储中间文件。如果为None，使用输出文件的父目录
        skip_existing : bool
            是否跳过已存在的输出文件

        Returns
        -------
        dict
            输入文件到信息字典的映射
        """
        results = {}
        success_count = 0
        fail_count = 0

        for i, grib1_file in enumerate(grib1_files, 1):
            if logger:
                logger.info(f"[{i}/{len(grib1_files)}] 转换: {os.path.basename(grib1_file)}")

            basename = os.path.basename(grib1_file)
            nc_filename = basename.replace('.grib1', '.nc')
            output_nc_file = os.path.join(output_dir, nc_filename)

            # 检查是否跳过已存在的文件
            if skip_existing and os.path.exists(output_nc_file):
                if logger:
                    logger.info(f"输出文件已存在，跳过: {output_nc_file}")
                success_count += 1
                results[grib1_file] = {'status': 'skipped', 'message': '文件已存在'}
                continue

            try:
                success, info = CDOConverter.grib1_to_netcdf(
                    grib1_file, output_nc_file, logger, check_step_count, temp_dir
                )

                if success:
                    success_count += 1
                    results[grib1_file] = info
                else:
                    fail_count += 1
                    results[grib1_file] = info

            except Exception as e:
                error_msg = f"转换异常: {str(e)}"
                if logger:
                    logger.error(error_msg)
                fail_count += 1
                results[grib1_file] = {'error': error_msg}

        if logger:
            logger.info(f"转换完成: 成功 {success_count}，失败 {fail_count}")

        return results


def main_cli(args=None):
    """命令行接口"""
    parser = argparse.ArgumentParser(description='ECMWF GRIB1数据CDO转换工具')

    parser.add_argument('--input-file', type=str, default=None,
                        help='输入的GRIB1文件路径')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='输入目录（批量处理）')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--temp-dir', type=str, default=None,
                        help='临时文件夹路径（用于存储中间文件）')
    parser.add_argument('--no-check', action='store_true', default=False,
                        help='跳过时效数检查')
    parser.add_argument('--skip-existing', action='store_true', default=False,
                        help='跳过已存在的输出文件')
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

    # 检查参数
    if args.input_file is None and args.input_dir is None:
        parser.print_help()
        return 1

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    check_step_count = not args.no_check

    # 单个文件处理
    if args.input_file:
        if logger:
            logger.info(f"处理单个文件: {args.input_file}")

        nc_filename = os.path.basename(args.input_file).replace('.grib1', '.nc')
        output_nc_file = os.path.join(args.output_dir, nc_filename)

        if args.skip_existing and os.path.exists(output_nc_file):
            logger.info(f"输出文件已存在，跳过: {output_nc_file}")
            return 0

        success, info = CDOConverter.grib1_to_netcdf(
            args.input_file, output_nc_file, logger, check_step_count, args.temp_dir
        )

        if success:
            logger.info(f"✓ 转换成功: {output_nc_file}")
            return 0
        else:
            logger.error(f"✗ 转换失败: {info.get('error', 'Unknown error')}")
            return 1

    # 批量处理
    if args.input_dir:
        import glob

        pattern = os.path.join(args.input_dir, "*.grib1")
        grib1_files = sorted(glob.glob(pattern))

        if not grib1_files:
            logger.error(f"未找到GRIB1文件: {pattern}")
            return 1

        logger.info(f"找到 {len(grib1_files)} 个GRIB1文件")
        logger.info(f"检查时效数: {'否' if args.no_check else '是'}")

        results = CDOConverter.convert_multiple_files(
            grib1_files, args.output_dir, logger, check_step_count, args.temp_dir,
            args.skip_existing
        )

        success_count = sum(1 for info in results.values() if 'error' not in info)
        logger.info(f"批量转换完成: 成功 {success_count}/{len(grib1_files)}")

        return 0 if success_count == len(grib1_files) else 1


if __name__ == "__main__":
    sys.exit(main_cli())
