# EC_V5 使用说明

## 概述

EC_V5版本包含两个独立的处理模块：

1. **cdo_converter.py** - GRIB1格式转换为NetCDF格式
2. **ec_data_processor.py** - 时空降尺度处理（从NetCDF开始）

---

## cdo_converter.py 使用说明

### 单文件处理

将单个GRIB1文件转换为NetCDF格式：

```bash
python3 cdo_converter.py \
  --input-file /path/to/file.grib1 \
  --output-dir /path/to/output \
  --element TEM
```

**参数说明：**
- `--input-file`: 输入的GRIB1文件路径
- `--output-dir`: 输出目录（默认：nc_output）
- `--element`: 要素名称（2t, 10u, 10v, 100u, 100v, 10fg3, 2d, vis, tpe等）
- `--skip-existing`: 跳过已存在的文件（默认开启）
- `--no-skip-existing`: 不跳过已存在的文件

### 批量处理

处理目录中的所有GRIB1文件：

```bash
python3 cdo_converter.py \
  --input-dir /path/to/grib1_files \
  --output-dir /path/to/output
```

---

## ec_data_processor.py 使用说明

### 1. 单文件处理

处理单个NetCDF文件：

```bash
python3 ec_data_processor.py \
  --nc-file /path/to/file.nc \
  --output-dir /path/to/output \
  --element TEM \
  --base-time 2026032500
```

**参数说明：**
- `--nc-file`: 输入的NetCDF文件路径
- `--output-dir`: 输出目录（默认：1KM1H）
- `--element`: 要素名称
  - 标量: TEM, PRS, DPT, RH, PRE, VIS, GUST, MN2T6, MX2T6
  - 风场: WIND_10M, WIND_100M, WIND_60M
  - 风分量: 10U, 10V, 100U, 100V（仅用于批量处理）
- `--base-time`: 基准时间UTC，格式：YYYYMMDDHH
- `--output-filename`: 输出文件名（可选）
- `--save-micaps4`: 保存MICAPS4格式
- `--micaps4-output-dir`: MICAPS4输出目录
- `--skip-existing`: 跳过已存在的文件（默认开启）
- `--no-skip-existing`: 不跳过已存在的文件
- `--verbose`: 详细输出

### 2. 批量处理（推荐）

处理目录中的所有NC文件：

```bash
python3 ec_data_processor.py \
  --input-dir /path/to/nc_files \
  --output-dir /path/to/output \
  --save-micaps4 \
  --micaps4-output-dir /path/to/micaps
```

**批量处理特性：**
- 自动识别文件名中的要素类型
- 自动处理风场U/V合并为WIND文件
- 自动从温度数据生成相对湿度
- WIND_60M基于WIND_100M自动生成

### 3. 指定要素处理

只处理指定要素：

```bash
python3 ec_data_processor.py \
  --input-dir /path/to/nc_files \
  --output-dir /path/to/output \
  --element TEM
```

**只处理温度和风场：**

```bash
python3 ec_data_processor.py \
  --input-dir /path/to/nc_files \
  --output-dir /path/to/output \
  --elements 'TEM,WIND'
```

### 4. MICAPS4输出

保存MICAPS4格式文件：

```bash
python3 ec_data_processor.py \
  --input-dir /path/to/nc_files \
  --output-dir /path/to/output \
  --save-micaps4 \
  --micaps4-output-dir /path/to/micaps_output
```

**MICAPS4输出规则：**
- WIND_10M: 输出MICAPS4格式（type=11矢量格式）
- WIND_100M, WIND_60M: 不输出MICAPS4格式
- 其他标量要素: 输出MICAPS4格式（type=4标量格式）

### 5. 跳过已存在文件

默认跳过已存在的文件（节省处理时间）：

```bash
# 默认跳过（推荐）
python3 ec_data_processor.py --input-dir ... --output-dir ...

# 强制重新处理
python3 ec_data_processor.py --input-dir ... --output-dir ... --no-skip-existing
```

### 6. 相对湿度生成

自动从温度数据生成相对湿度：
- 使用简化公式：`RH = 50 + 30 * sin(温度/10)`
- 与温度文件相同的时空插值结果
- 如果温度文件不存在，跳过相对湿度生成

---

## 完整示例

### 标准批量处理流程

```bash
# 1. GRIB1转NetCDF
python3 cdo_converter.py \
  --input-dir /data/grib1 \
  --output-dir /data/nc_output

# 2. 数据降尺度 + MICAPS4输出
python3 ec_data_processor.py \
  --input-dir /data/nc_output \
  --output-dir /data/1KM1H \
  --save-micaps4 \
  --micaps4-output-dir /data/micaps \
  --verbose
```

### 只处理温度和风场

```bash
python3 ec_data_processor.py \
  --input-dir /data/nc_output \
  --output-dir /data/1KM1H \
  --elements 'TEM,WIND' \
  --save-micaps4 \
  --micaps4-output-dir /data/micaps
```

### 单文件处理（用于测试）

```bash
python3 ec_data_processor.py \
  --nc-file /data/nc_output/tem_2026032500.nc \
  --output-dir /data/1KM1H \
  --element TEM \
  --base-time 2026032500 \
  --save-micaps4 \
  --micaps4-output-dir /data/micaps \
  --verbose
```

---

## 输出文件命名规范

### NetCDF文件

格式：`{element}_0p01_1h_BJT_{time_str}.nc`

示例：
- `tem_0p01_1h_BJT_2026032508.nc`
- `wind_10m_0p01_1h_BJT_2026032508.nc`
- `rh_0p01_1h_BJT_2026032508.nc`

### MICAPS4文件

格式：`{model}_{element}_{time_str}.000`

示例：
- `ECMWF_TEM_260308.000` (北京时2026年03月08日，预报0小时)
- `ECMWF_WIND_260308.024` (预报24小时)
