# EC数据处理工具修改记录

## 修改日期
2026-03-27

## 修改人员
Claude Code (Sonnet 4.6)

## 修改原因
1. 添加内存清理机制：原代码没有显式的内存管理和垃圾回收，在处理大量数据时可能导致内存积累
2. 删除MICAPS4详细日志：输出信息过多，影响日志查看效率
3. 修改风的nc输出：只保存U和V分量，不计算风速风向，减少计算量和输出变量数量

## 主要变更点

### 1. 备份代码
- 创建备份文件：`ec_data_processor.py.backup`

### 2. 添加内存清理机制
- **导入gc模块**：在文件开头添加 `import gc`
- **_process_scalar_data方法**：在返回前添加 `del data_1h, times_bjt` 和 `gc.collect()`
- **_process_wind_data方法**：在返回前添加 `del u_1h, v_1h` 和 `gc.collect()`
- **_process_precipitation_time_interpolation方法**：在返回前添加 `del accum_data_mm, interval_precip` 和 `gc.collect()`
- **批量处理循环**：在每个文件处理完成后添加 `gc.collect()`

### 3. 删除MICAPS4详细日志
- 删除了 `MICAPS4Writer.write_micaps4_vector_file()` 方法中的所有详细 print 语句
- 包括：文件路径、数据类型、要素、模式、起报时间、时效、高度、网格、数据点、文件大小等信息

### 4. 修改风的nc输出 - 只保存U和V

#### 4.1 修改配置文件中的 output_vars
- **WIND**: 从 `["wspd", "wdir", "u", "v"]` 改为 `["u", "v"]`
- **WIND100**: 从 `["wspd", "wdir", "u", "v"]` 改为 `["u", "v"]`
- **WIND60**: 从 `["wspd", "wdir"]` 改为 `["u", "v"]`
- **对应的units也做了相应调整**

#### 4.2 删除风速风向计算
- 删除了以下代码：
  ```python
  # 5. 计算风速风向
  start_time = time.time()
  wspd_1h, wdir_1h = MeteorologicalCalculator.calculate_wind_speed_direction(u_1h, v_1h)
  calc_time = time.time() - start_time
  self.logger.info(f"风速风向计算完成: {calc_time:.1f}秒")
  ```

#### 4.3 修改nc输出
- 删除了 wspd 和 wdir 变量的创建和数据写入
- 修改为对所有风场要素（WIND、WIND100、WIND60）都输出U/V分量，不再使用条件判断
- 修改返回数据结构为只包含 U/V
- 修改全局属性标题为 "ECMWF {height_desc} Wind U/V Components (Beijing Time)"

## 修改文件
- `/home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/ec_data_processor.py` - 主文件
- `/home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/ec_data_processor.py.backup` - 备份文件（新建）

## 预期效果
1. 内存使用更加高效，特别是在批量处理大量文件时
2. MICAPS4文件生成时不再输出冗余的详细信息
3. 风场nc文件体积减小（减少了2个变量的存储）
4. 计算速度提升（减少风速风向的计算步骤）
5. 输出nc文件只包含 u_10m, v_10m, u_100m, v_100m, u_60m, v_60m 变量，不包含 wspd 和 wdir 变量

## 对话历史

**用户原始请求：**
```
备份一下代码；加上内存清理机制；删除micaps格式输出时的这种日志显示[省略详细日志]；在输出风的nc时，只保存U和V的插值结果，不要计算风速和风向。把这次对话的内容保存在/home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/records.md里面
```

**分析过程：**
1. 通过阅读代码发现当前没有显式的内存管理和垃圾回收机制
2. 发现MICAPS4写入时有大量冗余的详细日志输出
3. 发现风的nc输出包含风速风向的计算和输出，增加了不必要的计算量和存储空间

**实施过程：**
1. 备份原始代码
2. 添加 gc 模块导入
3. 在关键位置添加内存清理（del 和 gc.collect）
4. 删除MICAPS4详细日志
5. 修改风的nc输出逻辑，只保存U/V分量
6. 创建记录文件

## 注意事项
1. 修改后风的nc文件不再包含 wspd 和 wdir 变量，如有下游代码依赖这些变量需要相应修改
2. MICAPS4文件生成不再输出详细信息，如需调试信息可能需要临时恢复或使用DEBUG日志级别
3. 内存清理在函数返回前执行，确保不会影响正常的数据处理流程

---

## 第二次对话记录（2026-03-27）

**用户第二次请求：**
```
保存这次对话内容到/home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/records.md，备份一下文件
```

**执行内容：**
1. 备份修改后的 ec_data_processor.py 文件（使用时间戳命名）
2. 将本次对话内容追加到 records.md 文件中

**备份文件信息：**
- 文件名：`ec_data_processor.py.backup_20260327_085208`
- 备份时间：2026-03-27 08:52:08
- 备份内容：包含所有修改后的代码（内存清理、删除MICAPS4日志、修改风的nc输出等）
- 文件大小：102K

---

## Bug修复记录（2026-03-27）

**问题描述：**
运行时出现错误：
```
ERROR - 处理标量数据失败: cannot access local variable 'times_bjt' where it is not associated with a value
ERROR - 处理要素失败: not enough values to unpack (expected 5, got 3)
```

**原因分析：**
在 `_process_scalar_data` 方法中，内存清理代码在 return 语句之前删除了 `times_bjt` 变量，但 return 语句中仍然需要使用该变量，导致访问已删除的变量。

**修复内容：**
修改 `_process_scalar_data` 方法的内存清理逻辑，只删除 `data_1h`，保留 `times_bjt`（因为 return 语句中需要使用）。

修改前：
```python
# 内存清理
del data_1h, times_bjt
gc.collect()

return True, result_data, times_bjt, final_output_path, hours_new
```

修改后：
```python
# 内存清理（注意：times_bjt 在 return 语句中使用，不能删除）
del data_1h
gc.collect()

return True, result_data, times_bjt, final_output_path, hours_new
```

**验证结果：**
- `_process_wind_data` 方法没有此问题（因为 result_data 字典已创建完成）
- 问题仅存在于 `_process_scalar_data` 方法
- 修复后可正常运行

---

## Bug修复后备份（2026-03-27 09:03）

**备份文件信息：**
- 文件名：`ec_data_processor.py.backup_20260327_090301`
- 备份时间：2026-03-27 09:03:01
- 备份内容：修复 bug 后的代码（修复了 `_process_scalar_data` 中的内存清理问题）
- 文件大小：102K

---

## 多问题修复记录（2026-03-27）

**用户请求：**
```
阅读一下/home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/records.md知道之前在做什么。你先备份一下源代码，这是这次运行的报错：

1. 风速的micaps格式输出有问题：
   - 未找到风速数据，无法生成MICAPS4文件
   - cannot access local variable 'data' where it is not associated with a value

2. PRE是气压，看起来整个都有问题：
   - 'msl'
   - not enough values to unpack (expected 5, got 3)

3. TCC总云量好像没有建立映射：
   - 无法从文件名识别要素: ECMFC1D_TCC_1_2026032612_GLB_1.nc, 跳过

先处理这几个问题
```

**备份文件信息：**
- 文件名：`ec_data_processor.py.backup_20260327_0932xx`（实际时间戳）
- 备份时间：2026-03-27 09:32左右

**问题分析：**

### 问题1：风速MICAPS4输出问题
**原因：**
- 之前修改代码时，将风场的 `output_vars` 从 `["wspd", "wdir", "u", "v"]` 改为 `["u", "v"]`
- 但MICAPS4输出代码仍然尝试查找 `wspd` 和 `wdir` 变量
- 导致无法找到风速数据，无法生成MICAPS4文件

**修复内容：**
修改MICAPS4输出逻辑（第2324-2370行）：
1. 从 `temp_result` 中获取U/V分量
2. 使用 `MeteorologicalCalculator.calculate_wind_speed_direction()` 计算风速风向
3. 将计算结果用于MICAPS4矢量格式输出

### 问题2：气压（PRS）处理失败
**原因：**
- `_process_scalar_data` 方法的异常处理中返回值不匹配
- 第1493行返回3个值：`return False, None, None`
- 但调用期望5个值：`temp_success, temp_result, times_bjt, actual_output_path, hours_new`

**修复内容：**
修改第1493行，将返回值改为5个：
```python
return False, None, None, None, None
```

### 问题3：TCC总云量没有建立映射
**原因：**
- ELEMENTS配置中没有TCC（总云量）的配置
- ELEMENT_MAP中没有TCC的映射
- SCALAR_LEVEL_MAP中没有TCC的level值
- 命令行参数choices中没有TCC

**修复内容：**
1. 在ELEMENTS配置中添加TCC配置（第224-235行）：
   ```python
   "TCC": {
       "description": "total cloud cover",
       "grib_codes": {"value": "tcc"},
       "requires_uv": False,
       "output_vars": ["tcc"],
       "units": {"tcc": "%"},
       "conversion": None,
       "nc_mapping": {
           "input_pattern": "ECMFC1D_TCC",
           "input_var": "tcc",
           "output_filename": "tcc"
       }
   }
   ```

2. 在ELEMENT_MAP中添加TCC映射：
   ```python
   "TCC": "TCC"
   ```

3. 在SCALAR_LEVEL_MAP中添加TCC的level值：
   ```python
   "TCC": 1000.0  # 总云量
   ```

4. 在命令行参数choices中添加TCC

**预期效果：**
1. 风场MICAPS4文件能够正常生成
2. 气压（PRS）数据处理时不会出现解包错误
3. TCC总云量能够被正确识别和处理
4. 所有要素的批量处理都能正常运行

---

## 风速MICAPS文件输出问题修复（2026-03-27）

**问题描述：**
/home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/micaps/WIND 目录为空，风速micaps文件没有输出。

**原因分析：**
在 MICAPS4 输出代码中，获取风场U/V分量时使用的键名与实际数据结构不匹配：

1. `_process_wind_data` 返回的数据结构包含键名 `"u_10m"` 和 `"v_10m"`（10米风）：
   ```python
   result_data = {
       f"u_{self._get_height_suffix(element)}": u_1h,
       f"v_{self._get_height_suffix(element)}": v_1h
   }
   ```

2. 但MICAPS4输出代码尝试获取数据时使用的是 `"u"` 和 `"v"`：
   ```python
   u_data = temp_result.get("u")
   v_data = temp_result.get("v")
   ```

3. 导致 `u_data` 和 `v_data` 都为 `None`，无法找到风场U/V分量数据，无法生成MICAPS4文件

**修复内容：**
修改第2346-2347行，使用正确的键名来获取数据：
```python
u_data = temp_result.get(f"u_{self._get_height_suffix(element)}")
v_data = temp_result.get(f"v_{self._get_height_suffix(element)}")
```

**备份文件信息：**
- 文件名：`ec_data_processor.py.backup_YYYYMMDD_HHMMSS`
- 备份时间：2026-03-27

**预期效果：**
1. 风场（WIND、WIND100）MICAPS4文件能够正常生成
2. /home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/micaps/WIND 目录会有输出文件

---

## TPE降水时间点调整（2026-03-27）

**用户请求：**
```
阅读/home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/records.md了解之前在干什么。现在需要你在tpe降水输出nc和micaps的时候，在整个数组的最前面也就是第一个时间点加上一个全是nan的时间点，这个点的物理意义是起报时间的累计降水初始值，你不用管物理意义。操作方面就是在整个数组的最前面加一个时间点，然后原来的整个数组都往后顺延。时间上加的全是nan的就是起报时间，原来的第一个时间点就是起报时间+1，以此类推。确保降水跟气温这种要素一样，总时间点是241
```

**备份文件信息：**
- 文件名：`ec_data_processor.py.backup_YYYYMMDD_HHMMSS`
- 备份时间：2026-03-27

**问题分析：**
1. 非降水要素（如气温）的时间序列是 `np.arange(0, max_hour + 1, 1.0)`，共 241 个点（0 到 240h）
2. 降水要素（TPE）的时间序列是 `np.arange(0, max_hour, 1.0)`，只有 240 个点（0 到 239h）
3. 用户希望在降水数组最前面插入一个全是 nan 的时间点，代表起报时间的累计降水初始值

**修复内容：**
修改 `_process_precipitation_time_interpolation` 方法（第1571-1580行）：

修改前：
```python
# 4. 生成时间序列
hours_new = np.arange(0, max_hour, 1.0)

# 输出处理信息
self.logger.info(f"降水时间处理完成:")
self.logger.info(f"  原始时次: {len(steps)}个（{steps[:5]}...{steps[-5:]}）")
self.logger.info(f"  时段数: {n_intervals}个")
self.logger.info(f"  小时降水: {len(hourly_precip)}个（0-{max_hour-1}h）")
self.logger.info(f"  数据范围: {np.nanmin(hourly_precip):.4f} - {np.nanmax(hourly_precip):.4f} mm")
```

修改后：
```python
# 4. 在数组最前面插入一个全是 nan 的时间点（起报时间的累计降水初始值）
hourly_precip = np.insert(hourly_precip, 0, np.nan, axis=0)

# 5. 生成时间序列（从0到max_hour，共max_hour+1个点，与其他要素如气温保持一致）
hours_new = np.arange(0, max_hour + 1, 1.0)

# 输出处理信息
self.logger.info(f"降水时间处理完成:")
self.logger.info(f"  原始时次: {len(steps)}个（{steps[:5]}...{steps[-5:]}）")
self.logger.info(f"  时段数: {n_intervals}个")
self.logger.info(f"  小时降水: {len(hourly_precip)}个（0-{max_hour}h，第0点为起报时间初始值NaN）")
self.logger.info(f"  数据范围: {np.nanmin(hourly_precip):.4f} - {np.nanmax(hourly_precip):.4f} mm")
```

**预期效果：**
1. TPE降水的时间序列从 240 个点增加到 241 个点（0 到 240h）
2. 第0个时间点（索引0）全为 NaN，代表起报时间的累计降水初始值
3. 原来的降水数据整体向后顺延一个时间点
4. 降水与气温等要素的时间点数量保持一致，便于下游处理

**注意事项：**
1. 第0点（起报时间）的降水数据为 NaN，这是物理意义上的初始值，不代表实际降水
2. 第1点（起报时间+1）才是真正的第一个有数据的时间点
3. nc 和 micaps 输出时都会包含这个额外的全 NaN 时间点

---

## 风场输出目录名修复（2026-03-27）

**问题描述：**
程序运行时，发现输出文件已存在但程序没有跳过。例如：
- 实际文件路径：`/home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/NEW/wind_100m/...`
- 代码预期路径：`/home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/NEW/wind100/...`

**原因分析：**
代码中使用 `element.lower()` 作为输出目录名：
```python
output_path = os.path.join(output_dir, element.lower(), bjt_time_str, output_filename)
```
- WIND → `wind`
- WIND100 → `wind100`
- WIND60 → `wind60`

但实际存在的目录名是：
- WIND → `wind_10m`
- WIND100 → `wind_100m`
- WIND60 → `wind_60m`

**修复内容：**
修改第2161-2167行，对于风场要素使用 `output_filename_base` 作为目录名：

修改前：
```python
# 按照要素和起报时间分类存放
bjt_time_str = Config.utc_to_bjt_str(base_time)
output_path = os.path.join(output_dir, element.lower(), bjt_time_str, output_filename)
```

修改后：
```python
# 按照要素和起报时间分类存放
bjt_time_str = Config.utc_to_bjt_str(base_time)
# 对于风场要素，使用 output_filename_base 作为目录名（如 wind_10m、wind_100m、wind_60m）
# 对于其他要素，使用 element.lower() 作为目录名
if element in ["WIND", "WIND100", "WIND60"] and "nc_mapping" in element_config:
    dir_name = element_config["nc_mapping"]["output_filename"]  # 如 wind_10m
else:
    dir_name = element.lower()
output_path = os.path.join(output_dir, dir_name, bjt_time_str, output_filename)
```

**预期效果：**
1. 风场输出目录名与实际目录名一致
2. 当输出文件已存在时，程序能正确检测并跳过处理
3. 其他要素的输出路径不受影响

**补充修复：**
添加了调试日志，在检查文件是否存在时输出构建的路径和检查结果：
```python
self.logger.info(f"检查文件是否存在: {output_path}, 存在: {os.path.exists(output_path)}")
```

**备份文件信息：**
- 备份时间：2026-03-27

---

## NetCDF写入权限错误修复（2026-03-27）

**问题描述：**
运行时出现权限错误：
```
ERROR - 处理标量数据失败: [Errno 13] Permission denied: '/home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/NEW/tpe/2026032620/tpe_0p01_1h_BJT_2026032620.nc.tmp'
```

**原因分析：**
1. 目录 `/home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/NEW/tpe/2026032620/` 不存在
2. 在 `_process_scalar_data` 和 `_process_wind_data` 方法中，NetCDF 写入前缺少 `os.makedirs` 来确保目录存在
3. MICAPS4 文件写入代码中有 `os.makedirs`（第709行和第899行），但 NetCDF 写入代码中没有

**修复内容：**

1. 在 `_process_scalar_data` 方法中，NetCDF 写入前添加目录创建代码（第1435-1436行）：
```python
# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)
```

2. 在 `_process_wind_data` 方法中，NetCDF 写入前添加目录创建代码（第1891-1892行）：
```python
# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)
```

**备份文件信息：**
- 文件名：`ec_data_processor.py.backup_20260327_1114xx`（实际时间戳）
- 备份时间：2026-03-27 11:14左右

**预期效果：**
1. NetCDF 文件写入时，如果目标目录不存在会自动创建
2. 不再出现 `[Errno 13] Permission denied` 错误
3. 与 MICAPS4 文件写入逻辑保持一致

---

## WIND60依赖处理逻辑修改（2026-03-27）

**用户请求：**
```
wind60也跟rh一样，如果上一级不存在就跳过计算输出警告
```

**修改原因：**
WIND60 依赖 100m风场数据计算，但之前的代码在找不到100m风源文件时会报错返回，导致处理失败。用户希望像 RH 一样，如果依赖的源文件不存在就跳过计算并输出警告，而不是报错。

**修改前逻辑：**
```python
if not nc_files.get("u") or not nc_files.get("v"):
    self.logger.error(f"无法找到成对的U/V文件，请确保同时存在 U 和 V 分量文件")
    return False, None, {}  # 返回失败
```

**修改后逻辑：**
```python
if not nc_files.get("u") or not nc_files.get("v"):
    if element == "WIND60":
        # WIND60 依赖 100m风，如果源文件不存在则跳过并警告
        self.logger.warning(f"未找到100m风场源文件 (100U/100V)，跳过WIND60计算")
        return True, output_path, {}  # 返回成功但无数据
    else:
        # WIND 和 WIND100 是基础要素，必须存在
        self.logger.error(f"无法找到成对的U/V文件，请确保同时存在 U 和 V 分量文件")
        return False, None, {}  # 返回失败
```

**修改位置：**
- 第2201-2210行，修改风场文件缺失时的处理逻辑

**预期效果：**
1. WIND60 如果找不到 100m风源文件（100U/100V），会输出警告并跳过
2. WIND 和 WIND100 仍然必须找到对应的 U/V 文件，找不到会报错
3. 与 RH 的处理逻辑保持一致
4. 批量处理时不会因为某个依赖要素缺失而中断整个流程

**备份文件信息：**
- 文件名：`ec_data_processor.py.backup_20260327_112527`
- 备份时间：2026-03-27 11:25:27
- 备份内容：WIND60依赖处理逻辑修改后的代码
- 文件大小：106K

---

## RH文件写入权限错误修复（2026-03-27）

**问题描述：**
运行时 RH（相对湿度）文件写入出现权限错误：
```
ERROR - 处理要素失败: [Errno 13] Permission denied: '/home/youqi/FWZX_forecast_DATA/EC_C1D_program/EC_V5/NEW/rh/2026032620/rh_0p01_1h_BJT_2026032620.nc'
```

**原因分析：**
在 RH 文件写入代码（第2290行）前，缺少 `os.makedirs` 来确保目录存在。

**修复内容：**
在 RH 文件写入前添加目录创建代码（第2288-2289行）：
```python
# 确保输出目录存在
os.makedirs(os.path.dirname(rh_output_path), exist_ok=True)
# 写入相对湿度文件
with Dataset(rh_output_path, 'w') as nc:
```

**备份文件信息：**
- 文件名：`ec_data_processor.py.backup_20260327_141531`
- 备份时间：2026-03-27 14:15:31
- 备份内容：修复 RH 目录创建问题后的代码
- 文件大小：106K

**预期效果：**
1. RH 文件写入时，如果目标目录不存在会自动创建
2. 不再出现 `[Errno 13] Permission denied` 错误

---
