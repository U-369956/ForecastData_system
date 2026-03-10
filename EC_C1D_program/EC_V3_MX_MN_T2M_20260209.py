#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECMWF 多要素数据业务化处理系统 - 添加MAXTEMP6和MINTEMP6支持
"""


'''
# 保存为 EC_V3_MX_MN_T2M.py，然后运行

# 单个文件处理
python EC_V3_MX_MN_T2M_20260209.py \
  --input-file /path/to/ECMFC1D_MX2T6_1_2026020900_GLB_1.grib1 \
  --output-dir /path/to/output \
  --element MAXTEMP6

# 批量处理目录
python EC_V3_MX_MN_T2M_20260209.py \
  --input-dir /home/youqi/FWZX_forecast_DATA/data_demo \
  --output-dir /home/youqi/FWZX_forecast_DATA/output \
  --elements 'MAXTEMP6,MINTEMP6' \
  --save-micaps4 \
  --micaps4-output-dir /home/youqi/FWZX_forecast_DATA/tonst

# 混合要素处理
python EC_V3_MX_MN_T2M_20260209.py \
  --input-dir /home/youqi/FWZX_forecast_DATA/data_demo \
  --output-dir /home/youqi/FWZX_forecast_DATA/output \
  --elements 'TEM,MAXTEMP6,MINTEMP6,WIND' \
  --save-micaps4
'''



import os
import sys
import logging
import argparse
import numpy as np
from datetime import datetime, timedelta
import warnings

# 导入原有模块
from EC_V3_NC_MICAPS_tmp临时文件_20260205 import (
    Config,
    MeteorologicalCalculator,
    MICAPS4Writer,
    setup_logger,
    DataCacheManager,
    ECProcessor,
    create_file_sets_from_directory,
    process_multiple_elements,
    process_single_element,
    process_element_directory
)

# ================= 配置更新 =================
# 更新Config类，添加MAXTEMP6和MINTEMP6要素
def update_config():
    """更新配置，添加新要素"""
    Config.ELEMENTS.update({
        "MAXTEMP6": {
            "description": "Maximum temperature in past 6 hours at 2m",
            "grib_codes": {"value": "MX2T6"},
            "requires_uv": False,
            "output_vars": ["maxtemp6"],
            "units": {"maxtemp6": "°C"},
            "conversion": "K_to_C",  # 开尔文转摄氏度
            "time_interval": "6h",  # 原始时间间隔
            "forecast_type": "accumulated"
        },
        "MINTEMP6": {
            "description": "Minimum temperature in past 6 hours at 2m",
            "grib_codes": {"value": "MN2T6"},
            "requires_uv": False,
            "output_vars": ["mintemp6"],
            "units": {"mintemp6": "°C"},
            "conversion": "K_to_C",  # 开尔文转摄氏度
            "time_interval": "6h",  # 原始时间间隔
            "forecast_type": "accumulated"
        }
    })
    
    # 更新MICAPS要素映射
    MICAPS4Writer.ELEMENT_MAP.update({
        "MAXTEMP6": "2MT_MAX6",  # 过去6小时最高气温
        "MINTEMP6": "2MT_MIN6",  # 过去6小时最低气温
    })

# ================= 增强的文件搜索函数 =================
def create_file_sets_for_maxtemp_mintemp(input_dir: str, element: str):
    """为MAXTEMP6和MINTEMP6创建文件集"""
    from glob import glob
    import os
    import re
    
    file_sets = []
    
    # 获取要素配置
    if element not in Config.ELEMENTS:
        print(f"错误: 不支持的元素 {element}")
        return []
    
    element_config = Config.ELEMENTS[element]
    grib_code = element_config["grib_codes"]["value"]
    
    # 尝试多种可能的文件名模式
    patterns = [
        f"*{grib_code}*.grib*",  # 原模式
        f"*{grib_code.lower()}*.grib*",  # 小写
        f"*{grib_code.upper()}*.grib*",  # 大写
    ]
    
    # 添加特定要素的文件名模式
    if element == "MAXTEMP6":
        patterns.extend([
            "*MX2T6*.grib*",
            "*mx2t6*.grib*",
            "*MAXT*.grib*",
            "*max*.grib*"
        ])
    elif element == "MINTEMP6":
        patterns.extend([
            "*MN2T6*.grib*",
            "*mn2t6*.grib*",
            "*MINT*.grib*",
            "*min*.grib*"
        ])
    
    # 查找所有匹配的文件
    all_files = []
    for pattern in patterns:
        files = glob(os.path.join(input_dir, pattern))
        all_files.extend(files)
    
    # 去重并排序
    all_files = sorted(set(all_files))
    
    for file_path in all_files:
        file_sets.append({"value": file_path})
        print(f"找到{element}文件: {os.path.basename(file_path)}")
    
    print(f"为要素 {element} 找到 {len(file_sets)} 个文件集")
    return file_sets

# ================= 批量处理函数 =================
def batch_process_enhanced(processor, element: str, input_dir: str, output_dir: str, 
                          skip_existing: bool = True, save_micaps4: bool = False,
                          micaps4_output_dir: str = None) -> dict:
    """
    批量处理目录中的文件
    
    Parameters
    ----------
    processor : ECProcessor
        处理器实例
    element : str
        要素名称
    input_dir : str
        输入目录
    output_dir : str
        输出目录
    skip_existing : bool
        是否跳过已存在的文件
    save_micaps4 : bool
        是否保存MICAPS4格式
    micaps4_output_dir : str
        MICAPS4输出目录
        
    Returns
    -------
    dict
        处理统计信息
    """
    logger = processor.logger
    
    # 创建文件集
    if element in ["MAXTEMP6", "MINTEMP6"]:
        file_sets = create_file_sets_for_maxtemp_mintemp(input_dir, element)
    else:
        file_sets = create_file_sets_from_directory(input_dir, element)
    
    if not file_sets:
        logger.error(f"在目录 {input_dir} 中未找到有效的文件集 - 要素: {element}")
        return {'total': 0, 'success': 0, 'failed': 0, 'micaps4_files': 0}
    
    logger.info(f"找到 {len(file_sets)} 个文件集")
    
    stats = {'total': 0, 'success': 0, 'failed': 0, 'micaps4_files': 0}
    
    for i, file_set in enumerate(file_sets):
        stats['total'] += 1
        logger.info(f"处理文件集 {i+1}/{len(file_sets)}: {list(file_set.values())[0]}")
        
        try:
            success, output_path, micaps4_files = processor.process_element(
                element=element,
                input_files=file_set,
                output_dir=output_dir,
                skip_existing=skip_existing,
                save_micaps4=save_micaps4,
                micaps4_output_dir=micaps4_output_dir,
                use_cached_temp_data=True
            )
            
            if success:
                stats['success'] += 1
                if micaps4_files:
                    micaps4_count = sum(len(files) for files in micaps4_files.values())
                    stats['micaps4_files'] += micaps4_count
                logger.info(f"✓ 文件集 {i+1} 处理成功")
            else:
                stats['failed'] += 1
                logger.error(f"✗ 文件集 {i+1} 处理失败")
                
        except Exception as e:
            stats['failed'] += 1
            logger.error(f"✗ 文件集 {i+1} 处理异常: {str(e)}")
    
    return stats

# ================= 命令行接口 =================
def main_cli_enhanced():
    """增强的命令行接口主函数"""
    parser = argparse.ArgumentParser(description='ECMWF多要素数据处理器 - 支持过去6小时最高/最低气温')
    
    # 基本参数
    parser.add_argument('--elements', type=str, default=None,
                       help='要素名称列表，用逗号分隔，如: TEM,MAXTEMP6,MINTEMP6,WIND')
    
    parser.add_argument('--element', type=str, default=None,
                       choices=['WIND', 'GUST', 'TEM', 'MAXTEMP6', 'MINTEMP6', 'PRS', 'DPT', 'RH'],
                       help='单个要素名称（与--elements互斥）')
    
    # 输入方式1: 单个文件/文件集
    parser.add_argument('--input-file', type=str, default=None,
                       help='输入文件路径（标量要素）')
    
    parser.add_argument('--input-u-file', type=str, default=None,
                       help='U分量文件路径（风场要素）')
    
    parser.add_argument('--input-v-file', type=str, default=None,
                       help='V分量文件路径（风场要素）')
    
    # 输入方式2: 目录批量处理
    parser.add_argument('--input-dir', type=str, default=None,
                       help='输入目录（包含要素文件）')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default=None,
                       help='NetCDF输出目录')
    
    parser.add_argument('--base-time', type=str, default=None,
                       help='基准时间（UTC），格式: YYYYMMDDHH')
    
    # MICAPS4参数
    parser.add_argument('--save-micaps4', action='store_true', default=False,
                       help='保存MICAPS4格式文件')
    
    parser.add_argument('--micaps4-output-dir', type=str, default=None,
                       help='MICAPS4输出目录')
    
    # 处理选项
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='跳过已存在的输出文件')
    
    parser.add_argument('--no-skip-existing', action='store_false', dest='skip_existing',
                       help='不跳过已存在的输出文件')
    
    # 配置参数
    parser.add_argument('--lon-west', type=float, default=110.0,
                       help='区域西边界经度')
    
    parser.add_argument('--lon-east', type=float, default=127.0,
                       help='区域东边界经度')
    
    parser.add_argument('--lat-south', type=float, default=34.0,
                       help='区域南边界纬度')
    
    parser.add_argument('--lat-north', type=float, default=44.0,
                       help='区域北边界纬度')
    
    parser.add_argument('--resolution', type=float, default=0.01,
                       help='输出分辨率（度）')
    
    # 缓存参数
    parser.add_argument('--cache-dir', type=str, default='/tmp/ecmwf_cache',
                       help='缓存目录')
    
    parser.add_argument('--clear-cache', action='store_true', default=False,
                       help='清空缓存')
    
    args = parser.parse_args()
    
    # 检查要素参数
    if args.elements is None and args.element is None:
        print("错误: 必须提供--element或--elements参数")
        parser.print_help()
        return 1
    
    if args.elements is not None and args.element is not None:
        print("错误: --element和--elements参数不能同时使用")
        return 1
    
    # 解析要素列表
    if args.elements:
        # 解析逗号分隔的要素列表
        elements = [elem.strip().upper() for elem in args.elements.split(',')]
        # 检查要素是否有效
        valid_elements = ['WIND', 'GUST', 'TEM', 'MAXTEMP6', 'MINTEMP6', 'PRS', 'DPT', 'RH']
        for elem in elements:
            if elem not in valid_elements:
                print(f"错误: 无效的要素名称 '{elem}'，有效要素: {valid_elements}")
                return 1
    else:
        # 单个要素
        elements = [args.element]
    
    # 检查输入参数
    if args.input_file is None and args.input_dir is None and (args.input_u_file is None or args.input_v_file is None):
        if len(elements) == 1 and elements[0] == "WIND" and (args.input_u_file is None or args.input_v_file is None):
            print("错误: 风场要素需要提供U和V分量文件，或输入目录")
            parser.print_help()
            return 1
        elif len(elements) > 1 and args.input_dir is None:
            print("错误: 处理多个要素必须提供输入目录")
            parser.print_help()
            return 1
    
    # 解析基准时间
    base_time = None
    if args.base_time:
        try:
            base_time = datetime.strptime(args.base_time, "%Y%m%d%H")
        except ValueError:
            print(f"错误: 无法解析基准时间 {args.base_time}")
            return 1
    
    # 更新配置（添加新要素）
    update_config()
    
    # 更新区域配置
    Config.REGION = {
        "lon_w": args.lon_west,
        "lon_e": args.lon_east,
        "lat_s": args.lat_south,
        "lat_n": args.lat_north
    }
    Config.RESOLUTION = args.resolution
    
    # 设置日志
    logger = setup_logger()
    logger.info(f"{'='*80}")
    logger.info(f"ECMWF多要素数据处理器 - 支持过去6小时最高/最低气温")
    logger.info(f"要素列表: {', '.join(elements)}")
    logger.info(f"区域范围: {args.lon_west}E - {args.lon_east}E, {args.lat_south}N - {args.lat_north}N")
    logger.info(f"分辨率: {args.resolution}度 (约1km)")
    logger.info(f"保存MICAPS4格式: {'是' if args.save_micaps4 else '否'}")
    if args.clear_cache:
        logger.info("将清空缓存")
    logger.info(f"{'='*80}")
    
    # 创建缓存管理器
    cache_manager = DataCacheManager(cache_dir=args.cache_dir)
    
    # 清空缓存（如果需要）
    if args.clear_cache:
        cache_manager.clear_cache()
        logger.info("缓存已清空")
    
    # 创建处理器
    processor = ECProcessor(
        logger=logger, 
        save_micaps4=args.save_micaps4,
        micaps4_output_dir=args.micaps4_output_dir,
        cache_manager=cache_manager
    )
    
    total_stats = {
        'total_elements': 0,
        'success_elements': 0,
        'failed_elements': 0,
        'total_filesets': 0,
        'success_filesets': 0,
        'failed_filesets': 0,
        'total_micaps4_files': 0
    }
    
    try:
        # 循环处理每个要素
        for element in elements:
            total_stats['total_elements'] += 1
            logger.info(f"\n{'='*80}")
            logger.info(f"开始处理要素 [{total_stats['total_elements']}/{len(elements)}]: {element}")
            logger.info(f"{'='*80}")
            
            element_success = True
            
            try:
                # 处理方式1: 单个文件/文件集（仅支持单个要素）
                if args.input_dir is None and len(elements) == 1:
                    # 构建输入文件字典
                    input_files = {}
                    
                    if element == "WIND":
                        if args.input_u_file and args.input_v_file:
                            input_files = {"u": args.input_u_file, "v": args.input_v_file}
                        else:
                            logger.error("风场要素需要U和V分量文件")
                            element_success = False
                            total_stats['failed_elements'] += 1
                            continue
                    else:
                        # 标量要素，包括MAXTEMP6和MINTEMP6
                        if args.input_file:
                            input_files = {"value": args.input_file}
                        else:
                            logger.error(f"{element}要素需要输入文件")
                            element_success = False
                            total_stats['failed_elements'] += 1
                            continue
                    
                    # 处理单个文件集
                    success, output_path, micaps4_files = processor.process_element(
                        element=element,
                        input_files=input_files,
                        output_dir=args.output_dir,
                        base_time=base_time,
                        skip_existing=args.skip_existing,
                        save_micaps4=args.save_micaps4,
                        micaps4_output_dir=args.micaps4_output_dir,
                        use_cached_temp_data=True
                    )
                    
                    if success:
                        logger.info(f"要素 {element} 处理成功")
                        logger.info(f"NetCDF文件: {output_path}")
                        if micaps4_files:
                            micaps4_count = sum(len(files) for files in micaps4_files.values())
                            logger.info(f"MICAPS4文件: {micaps4_count} 个")
                            total_stats['total_micaps4_files'] += micaps4_count
                        total_stats['success_elements'] += 1
                    else:
                        logger.error(f"要素 {element} 处理失败")
                        element_success = False
                        total_stats['failed_elements'] += 1
                
                # 处理方式2: 目录批量处理
                else:
                    if args.input_dir is None:
                        logger.error(f"处理要素 {element} 需要输入目录")
                        element_success = False
                        total_stats['failed_elements'] += 1
                        continue
                    
                    logger.info(f"批量处理目录: {args.input_dir} - 要素: {element}")
                    
                    # 使用增强的批量处理函数
                    stats = batch_process_enhanced(
                        processor=processor,
                        element=element,
                        input_dir=args.input_dir,
                        output_dir=args.output_dir,
                        skip_existing=args.skip_existing,
                        save_micaps4=args.save_micaps4,
                        micaps4_output_dir=args.micaps4_output_dir
                    )
                    
                    # 更新统计
                    total_stats['total_filesets'] += stats['total']
                    total_stats['success_filesets'] += stats['success']
                    total_stats['failed_filesets'] += stats['failed']
                    total_stats['total_micaps4_files'] += stats['micaps4_files']
                    
                    # 记录处理结果
                    if stats['failed'] > 0:
                        logger.warning(f"要素 {element} 处理有 {stats['failed']} 个文件集失败")
                        element_success = False
                    else:
                        logger.info(f"要素 {element} 处理全部成功")
                    
                    if element_success:
                        total_stats['success_elements'] += 1
                    else:
                        total_stats['failed_elements'] += 1
            
            except Exception as e:
                logger.error(f"处理要素 {element} 时发生异常: {str(e)}", exc_info=True)
                total_stats['failed_elements'] += 1
                continue  # 继续处理下一个要素
        
        # 打印总体统计
        logger.info(f"\n{'='*80}")
        logger.info("所有要素处理完成！")
        logger.info(f"{'='*80}")
        logger.info(f"要素处理统计:")
        logger.info(f"  总要素数: {total_stats['total_elements']}")
        logger.info(f"  成功要素: {total_stats['success_elements']}")
        logger.info(f"  失败要素: {total_stats['failed_elements']}")
        
        if total_stats['total_filesets'] > 0:
            logger.info(f"文件集处理统计:")
            logger.info(f"  总文件集数: {total_stats['total_filesets']}")
            logger.info(f"  成功文件集: {total_stats['success_filesets']}")
            logger.info(f"  失败文件集: {total_stats['failed_filesets']}")
            success_rate = (total_stats['success_filesets'] / total_stats['total_filesets'] * 100) if total_stats['total_filesets'] > 0 else 0
            logger.info(f"  成功率: {success_rate:.1f}%")
        
        if args.save_micaps4:
            logger.info(f"  总MICAPS4文件数: {total_stats['total_micaps4_files']}")
        
        logger.info(f"{'='*80}")
        
        # 返回状态码：如果有任何要素失败，返回1
        return 0 if total_stats['failed_elements'] == 0 else 1
    
    except Exception as e:
        logger.error(f"处理过程中发生严重错误: {str(e)}", exc_info=True)
        return 1


# ================= 简化调用接口 =================
def process_maxtemp_mintemp(input_dir: str, output_dir: str = None, 
                           save_micaps4: bool = False, **kwargs):
    """
    专门处理过去6小时最高/最低气温的简化接口
    
    Parameters
    ----------
    input_dir : str
        输入目录
    output_dir : str, optional
        输出目录
    save_micaps4 : bool
        是否保存MICAPS4格式
    **kwargs : dict
        其他参数
        
    Returns
    -------
    dict
        处理统计信息
    """
    update_config()  # 更新配置
    
    # 处理最高气温
    print("开始处理过去6小时最高气温(MAXTEMP6)...")
    maxtemp_stats = process_element_directory(
        element="MAXTEMP6",
        input_dir=input_dir,
        output_dir=output_dir,
        save_micaps4=save_micaps4,
        **kwargs
    )
    
    # 处理最低气温
    print("\n开始处理过去6小时最低气温(MINTEMP6)...")
    mintemp_stats = process_element_directory(
        element="MINTEMP6",
        input_dir=input_dir,
        output_dir=output_dir,
        save_micaps4=save_micaps4,
        **kwargs
    )
    
    return {
        'MAXTEMP6': maxtemp_stats,
        'MINTEMP6': mintemp_stats
    }


if __name__ == "__main__":
    # 抑制警告
    warnings.filterwarnings("ignore")
    
    # 运行命令行接口
    sys.exit(main_cli_enhanced())