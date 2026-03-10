#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
读取 ECMWF 10 m U/V grib1（多 step），
按经纬度范围裁剪并插值到 0.01° 规则网格，
最终生成 (time, lat, lon) 维度的 NetCDF。
"""
# ================= 用户参数 =================
data_dir = "/home/youqi/FWZX_forecast_DATA/data_demo"
u_file   = "ECMFC1D_10U_1_2026010400_GLB_1.grib1"
v_file   = "ECMFC1D_10V_1_2026010400_GLB_1.grib1"

lon_w, lon_e = 110.0, 121.0          # 目标经度
lat_s, lat_n = 35.0,  44.0           # 目标纬度
out_res      = 0.01                  # 输出分辨率

out_u_nc = "uwnd_10m_0p01_allSteps.nc"
out_v_nc = "vwnd_10m_0p01_allSteps.nc"
# ==========================================

import os
import numpy as np
import pygrib
from scipy.interpolate import RegularGridInterpolator as rgi
from netCDF4 import Dataset, date2num, num2date
from datetime import datetime, timedelta

def list_all_steps(fname):
    """返回 grib 里所有 step（小时）列表，按出现顺序"""
    grbs = pygrib.open(os.path.join(data_dir, fname))
    steps = [msg.step for msg in grbs]
    grbs.close()
    return sorted(list(set(steps)))   # 去重并升序

def read_one_step(fname, step):
    """读取指定 step 的原始 lat/lon/data"""
    grbs = pygrib.open(os.path.join(data_dir, fname))
    msg  = grbs.select(step=step)[0]   # 取第一条匹配
    data, lats, lons = msg.data()      # shape=(1441,2880)
    grbs.close()
    lat1d = lats[:, 0]                 # 通常降序
    lon1d = lons[0, :]
    return lat1d, lon1d, data

def crop_and_interp(lat_in, lon_in, data_in,
                    lat_out, lon_out):
    """双线性插值到目标网格"""
    if lat_in[-1] < lat_in[0]:        # 确保 lat 升序
        lat_in = lat_in[::-1]
        data_in = data_in[::-1, :]
    f = rgi((lat_in, lon_in), data_in,
            bounds_error=False, fill_value=None)
    lon2d, lat2d = np.meshgrid(lon_out, lat_out)
    return f((lat2d, lon2d))

def build_target_grid():
    """生成目标 0.01° 网格"""
    lon_out = np.arange(lon_w, lon_e + out_res/2, out_res)
    lat_out = np.arange(lat_n, lat_s - out_res/2, -out_res)  # 降序
    return lat_out, lon_out

def process_one_field(infile, outfile, varname):
    """完整流程：读取所有 step → 裁剪插值 → 写 NetCDF"""
    steps = list_all_steps(infile)
    print(f'{infile} 中共 {len(steps)} 个 step：{steps}')

    # 先拿一条消息拼出目标网格
    lat_src, lon_src, _ = read_one_step(infile, steps[0])
    lat_dst, lon_dst    = build_target_grid()

    # 创建 NetCDF
    with Dataset(outfile, 'w') as nc:
        nc.createDimension('time', len(steps))
        nc.createDimension('lat',  lat_dst.size)
        nc.createDimension('lon',  lon_dst.size)

        time           = nc.createVariable('time', 'i4', ('time',))
        lat_var        = nc.createVariable('lat',  'f4', ('lat',))
        lon_var        = nc.createVariable('lon',  'f4', ('lon',))
        data_var       = nc.createVariable(varname,'f4',('time','lat','lon'))

        lat_var[:]     = lat_dst
        lon_var[:]     = lon_dst
        lat_var.units  = 'degrees_north'
        lon_var.units  = 'degrees_east'

        # 时间坐标：用第一条消息的 validityDate/Time 推算
        grbs   = pygrib.open(os.path.join(data_dir, infile))
        baseMsg= grbs.select(step=steps[0])[0]
        baseDateTime = datetime.strptime(
            f"{baseMsg.validityDate:08d}{baseMsg.validityTime:04d}", "%Y%m%d%H%M")
        grbs.close()

        times = [baseDateTime + timedelta(hours=int(step)) for step in steps]
        time.units = 'hours since 1970-01-01 00:00:00'
        time.calendar = 'gregorian'
        time[:]    = date2num(times, units=time.units, calendar=time.calendar)

        # 逐 step 处理
        for i, step in enumerate(steps):
            print(f'processing step {step}h ...')
            _, _, data_src = read_one_step(infile, step)
            data_dst = crop_and_interp(lat_src, lon_src, data_src,
                                       lat_dst, lon_dst)
            data_var[i, ...] = data_dst

        data_var.units = 'm s-1'
    print(f'Saved {outfile}  (time, lat, lon) = {len(steps), lat_dst.size, lon_dst.size}')

# 执行
process_one_field(u_file, out_u_nc, 'u10')
process_one_field(v_file, out_v_nc, 'v10')