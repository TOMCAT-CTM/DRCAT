import pandas as pd
import numpy as np
import xarray as xr
import os
import argparse

def vmr_to_number_density(vmr, pressure_pa, temperature_k):
    """
    将体积混合比 (VMR) 转换为数密度 (molecules/cm^3)。

    Args:
        vmr (np.ndarray): 物种的体积混合比 (无单位分数)。
        pressure_pa (np.ndarray): 大气压，单位为帕斯卡 (Pa)。
        temperature_k (np.ndarray): 大气温度，单位为开尔文 (K)。

    Returns:
        np.ndarray: 数密度，单位为 molecules/cm^3。
    """
    k_boltzmann = 1.380649e-23  # J/K
    vmr = np.asarray(vmr)
    pressure_pa = np.asarray(pressure_pa)
    temperature_k = np.asarray(temperature_k)

    valid_temp_mask = temperature_k > 0
    number_density_cm3 = np.full(vmr.shape, np.nan)

    # n_air = P / (k * T) -> molecules/m^3
    n_air_valid = pressure_pa[valid_temp_mask] / (k_boltzmann * temperature_k[valid_temp_mask])

    # n_species = VMR * n_air -> molecules/m^3
    number_density_m3 = vmr[valid_temp_mask] * n_air_valid

    # Convert from molecules/m^3 to molecules/cm^3 (1 m^3 = 1e6 cm^3)
    number_density_cm3[valid_temp_mask] = number_density_m3 * 1e-6

    return number_density_cm3

def create_gridded_netcdf(df_with_preds, var_name, p_levels, output_dir, lat_res=2.78, lon_res=2.8125):
    """
    将包含预测结果的 long-format DataFrame 转换为格点化的 NetCDF 文件。
    """
    print(f"正在为变量 '{var_name}' 创建格点化 NetCDF 文件...")

    # --- 1. 定义网格 ---
    year = df_with_preds['Year'].iloc[0]
    days_in_year = 366 if pd.to_datetime(f'{year}-12-31').dayofyear == 366 else 365
    lats_grid = np.arange(-90, 90 + lat_res, lat_res)
    lons_grid = np.arange(0, 360, lon_res)
    doy_grid = np.arange(1, days_in_year + 1)

    # --- 2. 将经纬度和时间映射到网格索引 ---
    df_grid = df_with_preds.copy()
    df_grid['doy'] = pd.to_datetime(df_grid['Time']).dt.dayofyear
    df_grid['lat_idx'] = pd.cut(df_grid['lat'], bins=lats_grid, labels=False, right=False)
    df_grid['lon_idx'] = pd.cut(df_grid['lon'], bins=lons_grid, labels=False, right=False)

    # --- 3. 高效聚合数据 ---
    aggregated_data = df_grid.groupby(['doy', 'pressure', 'lat_idx', 'lon_idx'])[var_name].mean()

    # --- 4. 创建空的4D数组并填充数据 ---
    output_array = np.full((len(doy_grid), len(p_levels), len(lats_grid)-1, len(lons_grid)-1), np.nan)

    # 从聚合结果中获取索引和值
    doy_indices_raw = aggregated_data.index.get_level_values('doy').to_numpy() - 1
    pressure_values = aggregated_data.index.get_level_values('pressure').to_numpy()
    lat_indices_raw = aggregated_data.index.get_level_values('lat_idx').to_numpy()
    lon_indices_raw = aggregated_data.index.get_level_values('lon_idx').to_numpy()
    values_raw = aggregated_data.to_numpy()

    # 使用searchsorted进行鲁棒的pressure匹配（处理浮点数舍入问题）
    p_levels_sorted = np.array(p_levels)
    pressure_indices = np.searchsorted(p_levels_sorted, pressure_values, side='left')
    
    # 验证匹配的有效性（确保找到的索引有效且值接近）
    valid_mask = np.ones(len(pressure_values), dtype=bool)
    tolerance = 1e-5  # 允许的浮点数误差范围
    
    for i, (p_actual, p_idx) in enumerate(zip(pressure_values, pressure_indices)):
        # 检查索引是否在范围内
        if p_idx >= len(p_levels_sorted):
            # 尝试最后一个索引
            p_idx = len(p_levels_sorted) - 1
            pressure_indices[i] = p_idx
        
        # 检查匹配值是否足够接近
        if abs(p_levels_sorted[p_idx] - p_actual) > tolerance:
            # 尝试也检查前一个索引
            if p_idx > 0 and abs(p_levels_sorted[p_idx - 1] - p_actual) < abs(p_levels_sorted[p_idx] - p_actual):
                pressure_indices[i] = p_idx - 1
            else:
                # 如果都不匹配，标记为无效
                print(f"Warning: Pressure {p_actual} does not match any level in {p_levels}. Closest: {p_levels_sorted[p_idx]}")
                valid_mask[i] = False

    # 使用有效掩码过滤所有数组
    doy_indices = doy_indices_raw[valid_mask].astype(int)
    pressure_indices = pressure_indices[valid_mask].astype(int)
    lat_indices = lat_indices_raw[valid_mask].astype(int)
    lon_indices = lon_indices_raw[valid_mask].astype(int)
    values = values_raw[valid_mask]
    
    print(f"Mapped {len(values)} out of {len(pressure_values)} data points successfully")

    # 使用清洗后的整数索引进行赋值
    output_array[doy_indices, pressure_indices, lat_indices, lon_indices] = values

    # --- 5. 创建并保存 xarray.DataArray ---
    da = xr.DataArray(
        data=output_array,
        dims=['doy', 'pressure', 'lat', 'lon'],
        coords={
            'doy': doy_grid,
            'pressure': p_levels,
            'lat': lats_grid[:-1] + lat_res / 2,  # 使用箱的中心作为坐标
            'lon': lons_grid[:-1] + lon_res / 2
        },
        name=var_name
    )
    da.attrs['long_name'] = f'Predicted {var_name.replace("_density", "")} Number Density'
    da.attrs['units'] = 'molecules cm-3'
    da.pressure.attrs['units'] = 'hPa'
    output_path = os.path.join(output_dir, f'{var_name}_gridded_{year}.nc')
    da.to_netcdf(output_path)
    print(f"✅ 已成功保存到: {output_path}")

if __name__ == "__main__":

    # python create_gridded_data.py --input_file base_data/ml_ready_OH_data_FE_augmented_2022.parquet --p_levels_file base/pressure_levels_OH.npy --plot_dir ML_OH/ML_OH_new/test --vars_to_save H2O
    
    for year in range(2005, 2015):
        for vars_to_save in ['TOMCAT_OH_density','true_MLS_OH_density','H2O']:  #
            print(f"\n--- Processing year: {year} ---")
            input_file = f'base_data/ml_ready_OH_data_FE_augmented_{year}.parquet'
            p_levels_file = 'pressure_levels_OH.npy'
            plot_dir = f'constant_var_data'
            

            # 构建命令行参数
            args = argparse.Namespace(
                input_file=input_file,
                p_levels_file=p_levels_file,
                plot_dir=plot_dir,
                vars_to_save=vars_to_save
            )


            # Load the results DataFrame
            try:
                results_df = pd.read_parquet(args.input_file)
                print(f"Loaded data from {args.input_file}")
            except Exception as e:
                print(f"Error loading input file: {e}")
                exit(1)

            # Load pressure levels from .npy file
            try:
                p_levels_full = np.load(args.p_levels_file)
                print(f"Loaded pressure levels from {args.p_levels_file}")
            except Exception as e:
                print(f"Error loading pressure levels file: {e}")
                exit(1)

            # Set min and max range for pressure levels
            plev_min = 1
            plev_max = 33
            p_levels = sorted([p for p in p_levels_full if plev_min <= p <= plev_max])
            print(f"Filtered and sorted pressure levels (min: {plev_min}, max: {plev_max}): {p_levels}")

            # Parse variables to save
            vars_to_save = [x.strip() for x in args.vars_to_save.split(',')]
            print(f"Variables to save: {vars_to_save}")

            # Ensure output directory exists
            os.makedirs(args.plot_dir, exist_ok=True)

            # --- 步骤 5: 保存为 NetCDF 文件 ---
            print("正在将所有处理后的变量保存为 NetCDF 文件...")
            
            # 定义变量转换规则：映射 (输出变量名) -> (输入列名或转换方式)
            var_conversion_rules = {
                'predicted_SSA_OH_density': ('predicted_SSA_OH_vmr', 'vmr_convert'),
                'predicted_MLS_OH_density': ('predicted_MLS_OH_vmr', 'vmr_convert'),
                'predicted_MLS_OH_uncertainty_density': ('predicted_MLS_OH_uncertainty_vmr', 'vmr_convert'),
                'calculated_SSA_OH_density': ('OH', 'vmr_convert'),
                'true_MLS_OH_density': ('MLS_OH', 'vmr_convert'),
                'TOMCAT_OH_density': ('TOMCAT_OH', 'vmr_convert'),
                'H2O_density': ('H2O', 'vmr_convert'),
                'H2O': ('H2O', 'direct'),  # H2O 直接保存，不转换
                'TOMCAT_OH': ('TOMCAT_OH', 'direct'),
            }
            
            for var_name in vars_to_save:
                # 检查是否在转换规则中
                if var_name in var_conversion_rules:
                    source_col, conversion_type = var_conversion_rules[var_name]
                    
                    # 如果源列不存在，跳过
                    if source_col not in results_df.columns:
                        print(f"  - 警告: 源列 '{source_col}' 不存在，跳过 '{var_name}'。")
                        continue
                    
                    # 根据转换类型执行操作
                    if conversion_type == 'vmr_convert' and source_col in results_df.columns:
                        print(f"正在从 {source_col} (VMR) 转换为 {var_name} (数密度)...")
                        pressure_pa = results_df['pressure'].values * 100  # hPa to Pa
                        temperature_k = results_df['Temperature'].values
                        results_df[var_name] = vmr_to_number_density(results_df[source_col], pressure_pa, temperature_k)
                    elif conversion_type == 'direct':
                        # 直接使用源列，不需要转换
                        print(f"使用原列 '{source_col}' 保存为 '{var_name}'...")
                        results_df[var_name] = results_df[source_col]
                else:
                    # 对于未在规则中的变量，检查是否直接存在
                    if var_name in results_df.columns:
                        print(f"使用现有列 '{var_name}' 直接保存...")
                    else:
                        print(f"  - 警告: 变量 '{var_name}' 不在转换规则中且不存在原列，跳过。")
                        continue
                
                # 最后检查变量是否有效，然后保存
                if var_name in results_df.columns.values and not results_df[var_name].isnull().all():
                    create_gridded_netcdf(results_df, var_name, p_levels, args.plot_dir)
                else:
                    print(f"  - 警告: 变量 '{var_name}' 无效或全为空值，跳过保存。")

            print("--- NetCDF 文件创建完成 ---")