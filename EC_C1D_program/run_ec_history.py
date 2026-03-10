#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECMWF 10 m U/V → 批量历史 → 裁剪插值 0.01° → 风速风向 → 1 h 线性 → UTC+8
一键整月运行，不调用任何外部单文件脚本
"""
import os
import numpy as np
from netCDF4 import Dataset, date2num
from datetime import datetime, timedelta
import pygrib
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import interp1d

# ========== 1. 用户参数（只改这里） ==========
ROOT_DIR   = "/mnt/CMADAAS/DATA/NAFP/ECMF/C1D"   # 根目录
START_DAY  = "20260101"                          # 含
END_DAY    = "20260101"                          # 含
OUT_DIR    = "./batch_output"                    # 输出目录
# 经纬度切片
LON_W, LON_E = 110.0, 121.0
LAT_S, LAT_N = 35.0,  44.0
OUT_RES      = 0.01
# ============================================

os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def date_range(start: str, end: str):
    s = datetime.strptime(start, "%Y%m%d")
    e = datetime.strptime(end, "%Y%m%d")
    for i in range((e - s).days + 1):
        yield (s + timedelta(days=i)).strftime("%Y%m%d")

def build_target_grid():
    lon_out = np.arange(LON_W, LON_E + OUT_RES/2, OUT_RES)
    lat_out = np.arange(LAT_N, LAT_S - OUT_RES/2, -OUT_RES)
    return lat_out, lon_out

def read_one_step(fpath, step, lat1, lat2, lon1, lon2):
    """带区域预裁剪的 grib 读取"""
    grbs = pygrib.open(fpath)
    msg  = grbs.select(step=step)[0]
    data, lats, lons = msg.data(lat1=lat1, lat2=lat2, lon1=lon1, lon2=lon2)
    grbs.close()
    lat1d = lats[:, 0]
    lon1d = lons[0, :]
    # 保证纬度降序
    if lat1d[-1] > lat1d[0]:
        lat1d  = lat1d[::-1]
        data   = data[::-1, :]
    return lat1d, lon1d, data.astype(np.float32)

def interp_splev(lat_in, lon_in, data_in, lat_out, lon_out):
    """RBS 二次样条 矢量化插值 - 保证升序"""
    # 1. 保证升序
    if lat_in[0] > lat_in[-1]:
        lat_in  = lat_in[::-1].copy()
        data_in = data_in[::-1, :].copy()
    # 2. 建样条
    spl = RBS(lat_in, lon_in, data_in, kx=2, ky=2, s=0)
    # 3. 矢量化求值
    lon2d, lat2d = np.meshgrid(lon_out, lat_out)
    return spl.ev(lat2d.ravel(), lon2d.ravel()).reshape(lat2d.shape).astype(np.float32)

def interp_to_1h(times_utc, data_3d):
    """1 h 线性时间插值"""
    t0 = times_utc[0]
    hours = np.array([(t - t0).total_seconds()/3600. for t in times_utc])
    hours_new = np.arange(hours[0], hours[-1]+1, 1)
    new_times_utc = [t0 + timedelta(hours=float(h)) for h in hours_new]
    f = interp1d(hours, data_3d, axis=0, kind='linear',
                 bounds_error=False, fill_value='extrapolate')
    data_new = f(hours_new).astype(np.float32)
    return new_times_utc, data_new

def process_one_cycle(day: str, hh: str):
    yyyymm = day[:6]
    in_dir = os.path.join(ROOT_DIR, yyyymm[:4], day)
    u_file = f"ECMFC1D_10U_1_{day}{hh}_GLB_1.grib1"
    v_file = f"ECMFC1D_10V_1_{day}{hh}_GLB_1.grib1"
    u_path = os.path.join(in_dir, u_file)
    v_path = os.path.join(in_dir, v_file)
    out_nc = os.path.join(OUT_DIR, f"wspd_wdir_10m_0p01_1h_BJT_{day}{hh}.nc")

    log(f"---- {day}{hh} ----")
    if not (os.path.exists(u_path) and os.path.exists(v_path)):
        log("[SKIP] 文件缺失"); return

    # 1. 扫描所有 step
    grbs = pygrib.open(u_path)
    steps = sorted({msg.step for msg in grbs})
    grbs.close()
    log(f"steps: {steps}")

    # 2. 建立目标网格
    lat_dst, lon_dst = build_target_grid()
    nt, nlat, nlon = len(steps), lat_dst.size, lon_dst.size
    u_cube = np.empty((nt, nlat, nlon), dtype=np.float32)
    v_cube = np.empty((nt, nlat, nlon), dtype=np.float32)

    # 3. 读 + 裁剪 + 插值
    for i, step in enumerate(steps):
        lat_src, lon_src, u_src = read_one_step(u_path, step, LAT_S, LAT_N, LON_W, LON_E)
        _       , _       , v_src = read_one_step(v_path, step, LAT_S, LAT_N, LON_W, LON_E)
        u_cube[i] = interp_splev(lat_src, lon_src, u_src, lat_dst, lon_dst)
        v_cube[i] = interp_splev(lat_src, lon_src, v_src, lat_dst, lon_dst)

    # 4. 算风速风向
    wspd_cube = np.sqrt(u_cube**2 + v_cube**2)
    wdir_cube = 180 + np.arctan2(u_cube, v_cube) * 180/np.pi
    wdir_cube = np.mod(wdir_cube, 360)

    # 5. 1 h 插值
    grbs = pygrib.open(u_path)
    base_msg = grbs.select(step=steps[0])[0]
    base_dt = datetime.strptime(
        f"{base_msg.validityDate:08d}{base_msg.validityTime:04d}", "%Y%m%d%H%M")
    grbs.close()
    times_utc = [base_dt + timedelta(hours=int(s)) for s in steps]
    times_utc, u_cube = interp_to_1h(times_utc, u_cube)
    _,         v_cube = interp_to_1h(times_utc, v_cube)
    _,    wspd1h = interp_to_1h(times_utc, wspd_cube)
    _,    wdir1h = interp_to_1h(times_utc, wdir_cube)

    # 6. 写 NC
    log(f"Writing {out_nc}")
    with Dataset(out_nc, 'w') as nc:
        nc.createDimension('time', len(times_utc))
        nc.createDimension('lat',  nlat)
        nc.createDimension('lon',  nlon)
        tvar = nc.createVariable('time', 'i4', ('time',))
        latv = nc.createVariable('lat',  'f4', ('lat',))
        lonv = nc.createVariable('lon',  'f4', ('lon',))
        wspd = nc.createVariable('wspd', 'f4', ('time','lat','lon'))
        wdir = nc.createVariable('wdir', 'f4', ('time','lat','lon'))
        latv[:] = lat_dst; latv.units = 'degrees_north'
        lonv[:] = lon_dst; lonv.units = 'degrees_east'
        tvar.units    = 'hours since 1970-01-01 00:00:00'
        tvar.calendar = 'gregorian'
        tvar[:]       = date2num([t + timedelta(hours=8) for t in times_utc],
                                 tvar.units, tvar.calendar)
        wspd[:] = wspd1h; wspd.units = 'm s-1'
        wdir[:] = wdir1h; wdir.units = 'degree'
    log(f"[SUCCESS] 保存完成 -> {out_nc}")

def main():
    log("========== 批量任务开始 ==========")
    for day in date_range(START_DAY, END_DAY):
        for hh in ("00", "12"):
            process_one_cycle(day, hh)
    log("========== 批量任务结束 ==========")

if __name__ == "__main__":
    main()