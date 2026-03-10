#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对已有 (time, lat, lon) 的 u10/v10/wspd/wdir 做 1 h 线性插值，
并把 UTC → 北京时（UTC+8）
"""
# ========== 用户参数 ==========
in_u  = "uwnd_10m_0p01_allSteps.nc"
in_v  = "vwnd_10m_0p01_allSteps.nc"
in_wd = "wspd_wdir_10m_0p01_allSteps.nc"   # 可选，如果已有 wspd/wdir

out_u  = "uwnd_10m_0p01_1h_BJT.nc"
out_v  = "vwnd_10m_0p01_1h_BJT.nc"
out_wd = "wspd_wdir_10m_0p01_1h_BJT.nc"
# ==============================

import numpy as np
from netCDF4 import Dataset, date2num, num2date
from scipy.interpolate import interp1d
from datetime import datetime, timedelta

BJT_OFFSET = timedelta(hours=8)

def load_time_series(nc_file, var_name):
    """返回 UTC 时间序列(1-D) 和 3-D 数组 (time,lat,lon)"""
    with Dataset(nc_file) as nc:
        t_var = nc.variables['time']
        utc_times = num2date(t_var[:], t_var.units, t_var.calendar)
        data = nc.variables[var_name][:]
        return utc_times, data, t_var.units, t_var.calendar, \
               nc.variables['lat'][:], nc.variables['lon'][:]

def interp_to_1h(times_utc, data_3d):
    """
    对每条网格点做线性插值 → 1 h 间隔
    times_utc: 1-D datetime64
    data_3d:   (nT, nLat, nLon)
    返回：new_times_utc(1-D), data_out(1-D→nT_new, nLat, nLon)
    """
    nT, nLat, nLon = data_3d.shape
    # 把 datetime 转为 "小时 since 第一个时刻" 做数值插值
    t0 = times_utc[0]
    hours = np.array([(t - t0).total_seconds() / 3600. for t in times_utc])
    # 1 h 间隔序列
    hours_new = np.arange(hours[0], hours[-1] + 1, 1)
    new_times_utc = [t0 + timedelta(hours=float(h)) for h in hours_new]

    # 预分配
    data_new = np.empty((hours_new.size, nLat, nLon), dtype=np.float32)

    # 逐网格点插值（scipy interp1d 自动沿 axis=0 线性插）
    f_interp = interp1d(hours, data_3d, axis=0, kind='linear',
                        bounds_error=False, fill_value='extrapolate')
    data_new[:] = f_interp(hours_new)
    return new_times_utc, data_new

def write_bjt_nc(outfile, var_name, data_bjt, lat, lon,
                 times_bjt, units, calendar):
    with Dataset(outfile, 'w') as nc:
        nc.createDimension('time', len(times_bjt))
        nc.createDimension('lat',  lat.size)
        nc.createDimension('lon',  lon.size)

        time_v = nc.createVariable('time', 'i4', ('time',))
        lat_v  = nc.createVariable('lat',  'f4', ('lat',))
        lon_v  = nc.createVariable('lon',  'f4', ('lon',))
        var_v  = nc.createVariable(var_name, 'f4', ('time', 'lat', 'lon'))

        lat_v[:] = lat;  lat_v.units  = 'degrees_north'
        lon_v[:] = lon;  lon_v.units  = 'degrees_east'

        # 北京时时间坐标
        time_v.units     = 'hours since 1970-01-01 00:00:00'
        time_v.calendar  = calendar
        time_v[:]        = date2num(times_bjt, time_v.units, calendar)

        var_v[:] = data_bjt
        var_v.units = 'm s-1' if 'spd' in var_name else 'degree'

def process_one_var(in_nc, var_name, out_nc):
    print(f'Processing {var_name} ...')
    times_utc, data_utc, tunits, tcal, lat, lon = load_time_series(in_nc, var_name)
    print('原始 UTC 范围：', times_utc[0], '→', times_utc[-1])
    times_1h_utc, data_1h = interp_to_1h(times_utc, data_utc)
    times_1h_bjt = [t + BJT_OFFSET for t in times_1h_utc]
    print('BJT 1h 范围： ', times_1h_bjt[0], '→', times_1h_bjt[-1])
    write_bjt_nc(out_nc, var_name, data_1h, lat, lon,
                 times_1h_bjt, tunits, tcal)
    print('Saved', out_nc)

# 依次处理 u/v/wspd/wdir
process_one_var(in_wd, 'wspd', out_wd.replace('wdir', 'wspd'))
process_one_var(in_wd, 'wdir', out_wd.replace('wspd', 'wdir'))