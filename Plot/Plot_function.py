import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from tqdm import tqdm
import joblib
import random
from sklearn.metrics import mean_squared_error, r2_score
from skimage.metrics import structural_similarity
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter 
import os
from tqdm import tqdm
from datetime import datetime
import matplotlib.font_manager as fm 
import time
# 在脚本开头确保已导入这些库
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import os
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable # <--- 1. 导入新工具



Constant_var_path = '/scratch/pdpv7239/constant_var/'



# 尝试导入 cartopy
try:
    import cartopy.crs as ccrs
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    print("警告: 未找到 'cartopy' 库。全球分布图将不包含地图特征。")



def log_sci_formatter(x, pos):
    """
    一个自定义的刻度格式化函数，用于生成 3x10^1 样式的标签。
    """
    # 计算指数 (例如，对于30，log10(30)约1.47，向下取整为1)
    exponent = np.floor(np.log10(x))
    # 计算尾数 (例如，30 / 10^1 = 3)
    mantissa = x / (10**exponent)
    
    # 如果尾数约等于1，则只显示10的指数形式
    if np.isclose(mantissa, 1.0):
        return f'$10^{{{int(exponent)}}}$'
    # 否则，显示完整的科学计数法
    else:
        return f'${int(mantissa)} \\times 10^{{{int(exponent)}}}$'

def convert_lon_to_0_360(da: xr.DataArray) -> xr.DataArray:
    """将xarray对象的经度坐标从[-180, 180]转换为[0, 360]。"""
    # 使用 assign_coords 进行干净的坐标更新
    da_converted = da.assign_coords(lon=(((da.lon + 360) % 360)))
    # 按新的经度坐标排序，这是一个好习惯
    da_sorted = da_converted.sortby('lon')
    return da_sorted

def perform_weighted_gridding(
    data_array: xr.DataArray,
    lat_spacing: float = 2.0,
    lon_spacing: float = 5.0,
    lat_half_width: float = 1.5,
    lon_half_width: float = 8.0,
    search_radius_factor: float = 2.0
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    在[0, 360]经度范围内，对稀疏数据进行加权网格化。
    """
    # print(f"开始执行加权网格化 (搜索半径因子: {search_radius_factor})...")
    start_time = time.time()

    stack = data_array.stack(points=('lon', 'lat')).dropna(dim='points')
    raw_lons = stack.lon.values
    raw_lats = stack.lat.values
    raw_values = stack.values

    if len(raw_values) == 0:
        print("错误：筛选后没有剩下任何有效数据点。")
        return None, None, None

    # print(f"网格化: {len(raw_values)} 个有效数据点。")

    # --- 修改点: 在 [0, 360] 范围内创建目标网格中心点 ---
    grid_lats_centers = np.arange(-90 + lat_spacing / 2, 90, lat_spacing)
    grid_lons_centers = np.arange(0 + lon_spacing / 2, 360, lon_spacing)
    gridded_data = np.full((len(grid_lats_centers), len(grid_lons_centers)), np.nan)
    
    lat_search_radius = search_radius_factor * lat_half_width
    lon_search_radius = search_radius_factor * lon_half_width
    
    # 核心的距离计算和加权平均算法无需改变，因为它是通用的
    for i, lat_g in enumerate(grid_lats_centers):
        for j, lon_g in enumerate(grid_lons_centers):
            d_lat = np.abs(raw_lats - lat_g)
            d_lon_abs = np.abs(raw_lons - lon_g)
            d_lon = np.minimum(d_lon_abs, 360 - d_lon_abs)
            mask = (d_lat < lat_search_radius) & (d_lon < lon_search_radius)

            if np.any(mask):
                nearby_values = raw_values[mask]
                nearby_d_lat = d_lat[mask]
                nearby_d_lon = d_lon[mask]
                weight_lat = (np.cos(np.pi / 2 * nearby_d_lat / lat_search_radius))**2
                weight_lon = (np.cos(np.pi / 2 * nearby_d_lon / lon_search_radius))**2
                total_weight = weight_lat * weight_lon
                weighted_sum = np.nansum(nearby_values * total_weight)
                sum_of_weights = np.nansum(total_weight)
                if sum_of_weights > 0:
                    gridded_data[i, j] = weighted_sum / sum_of_weights

    end_time = time.time()
    # print(f"网格化完成。耗时: {end_time - start_time:.2f} 秒。")
    
    return grid_lons_centers, grid_lats_centers, gridded_data


def _load_and_process_data(filepath, day_of_year):
    """
    加载单个NetCDF文件，选择特定日期的数据，并计算纬向平均值。
    """
    try:
        ds = xr.open_dataset(filepath)
        var_name = list(ds.data_vars)[0]
        daily_data = ds[var_name].sel(doy=day_of_year, method="nearest")
        if daily_data.notnull().any():
            return daily_data.mean(dim='lon', skipna=True)
    except (FileNotFoundError, IndexError, KeyError):
        # 捕获文件未找到、索引错误或键错误
        print(f"Info: 未能加载或处理文件 {os.path.basename(filepath)}。")
        return None

def _calculate_metrics(true_arr, pred_arr):
    """
    计算两个数组之间的RMSE、R²和SSIM指标。
    """
    # 确保数组形状一致
    if true_arr.shape != pred_arr.shape:
        return None

    # 为RMSE和R²找到共有的有效数据点
    valid_mask = np.isfinite(true_arr) & np.isfinite(pred_arr)
    if valid_mask.sum() < 2:  # 至少需要两个点才能计算R²
        return None
        
    v_true, v_pred = true_arr[valid_mask], pred_arr[valid_mask]
    rmse = np.sqrt(mean_squared_error(v_true, v_pred))
    r2 = r2_score(v_true, v_pred)

    # SSIM要求输入无NaN的数组
    true_no_nan = np.nan_to_num(true_arr)
    pred_no_nan = np.nan_to_num(pred_arr)
    
    # 使用真实数据的范围作为SSIM的data_range，这在比较中更具一致性
    data_range = true_no_nan.max() - true_no_nan.min()
    if data_range == 0:
        ssim = 1.0 if np.all(true_no_nan == pred_no_nan) else 0.0
    else:
        ssim = structural_similarity(true_no_nan, pred_no_nan, data_range=data_range)

    return {'rmse': rmse, 'r2': r2, 'ssim': ssim}

def _calculate_metrics_2(true_arr, pred_arr):
    """计算两个数组之间的RMSE和R²，处理NaN值。"""
    valid_mask = np.isfinite(true_arr) & np.isfinite(pred_arr)
    if valid_mask.sum() < 2:
        return {'rmse': np.nan, 'r2': np.nan}
    
    v_true, v_pred = true_arr[valid_mask], pred_arr[valid_mask]
    rmse = np.sqrt(mean_squared_error(v_true, v_pred))
    r2 = r2_score(v_true, v_pred)
    return {'rmse': rmse, 'r2': r2}



def plot_zonal_mean_timeline(variable_name, start_day, end_day, step_days, plot_dir, OUTPUT_DIR, year=None, plot_args=None):
    """
    绘制变量随时间变化的global zonal mean（绝对值）
    
    Parameters:
    -----------
    variable_name : str
        变量名称，如 'ML-Predicted OH'
    start_day : int
        起始天数
    end_day : int
        结束天数
    step_days : int
        步长（天）
    plot_dir : str
        图片保存目录
    OUTPUT_DIR : str
        包含输入 .nc 文件的数据目录
    year : int, optional
        年份，如果为None则使用默认年份
    plot_args : dict, optional
        其他绘图参数，如图形大小等
    """
    print(f"\n--- Generating Zonal Mean Timeline for {variable_name} ---")
    if plot_args is None:
        plot_args = {'fig_h': 10, 'fig_w': 10, 'cmap':'RdBu_r'}
    
    fig_h = plot_args.get('fig_h', 3)
    fig_w = plot_args.get('fig_w', 10)
    fig_cmap = plot_args.get('cmap', 'RdBu_r')


    if year is None:
        year = 2022  # 默认年份
    
    # 确定数据文件名
    if variable_name == 'ML-Predicted OH':
        fname = f'predicted_MLS_OH_density_gridded_{year}.nc'
    elif variable_name == 'MLS OH Obs':
        fname = f'true_MLS_OH_density_gridded_{year}.nc'
    elif variable_name == 'Calculated SSA-OH':
        fname = f'calculated_SSA_OH_density_gridded_{year}.nc'
    elif variable_name == 'Predicted SSA-OH':
        fname = f'predicted_SSA_OH_density_gridded_{year}.nc'
    elif variable_name == 'TOMCAT OH':
        fname = f'TOMCAT_OH_density_gridded_{year}.nc'
    else:
        print(f"Unknown variable name: {variable_name}")
        return
    
    try:
        ds = xr.open_dataset(os.path.join(OUTPUT_DIR, fname))
        var_name = list(ds.data_vars)[0]
    except FileNotFoundError:
        print(f"Could not load {fname}")
        return
    
    # 设置压力层范围
    plev_min = 0.9
    plev_max = 33
    
    # 生成时间序列
    days = range(start_day, end_day + 1, step_days)
    n_days = len(days)
    
    # 创建子图
    fig, axes = plt.subplots(nrows=n_days, ncols=1, figsize=(fig_w, fig_h*n_days), sharex=True, sharey=True)
    
    # 修正：确保 n_days=1 时 axes 也是一个列表
    if n_days == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # fig.suptitle(f'{variable_name} Global Zonal Mean Timeline ({year})', fontsize=16)
    
    # 收集所有数据用于确定颜色范围
    all_data = []
    for day in days:
        try:
            daily_data = ds[var_name].sel(doy=day, method="nearest")
            daily_data = daily_data.sel(pressure=slice(plev_min, plev_max))
            if daily_data.notnull().any():
                zonal_mean = daily_data.mean(dim='lon', skipna=True)
                all_data.append(zonal_mean)
        except (IndexError, KeyError):
            continue
    
    if not all_data:
        print("No data available to plot.")
        plt.close(fig) # 关闭未使用的图形
        return
    
    # 确定颜色范围
    max_val = max(d.max() for d in all_data)
    min_val = min(d.min() for d in all_data)
    
    # 绘制每个时间点的图
    for i, day in enumerate(days):
        try:
            daily_data = ds[var_name].sel(doy=day, method="nearest")
            daily_data = daily_data.sel(pressure=slice(plev_min, plev_max))
            if daily_data.notnull().any():
                zonal_mean = daily_data.mean(dim='lon', skipna=True)
                zonal_mean.plot.pcolormesh(ax=axes[i], x='lat', y='pressure', 
                                         cmap=fig_cmap, vmin=min_val, vmax=max_val,
                                         cbar_kwargs={'label': 'N. Density (molec cm⁻³)'})
                
                # --- 这是关键修改 ---
                # 2. 将 'day' (一年中的第几天) 和 'year' 转换为日期对象
                date_obj = datetime.strptime(f'{year} {day}', '%Y %j')
                
                # 3. 格式化日期字符串 (例如: "January 10")
                #    使用 .day 而不是 %d 来避免日期的前导零 (例如 "January 1" 而不是 "January 01")
                date_str = f"{date_obj.strftime('%B')} {date_obj.day}"
                
                # 4. 设置新的标题
                axes[i].set_title(f'Day {day} ({date_str})')
                # --- 修改结束 ---
                
                axes[i].set_yscale('log')
                axes[i].set_xlabel('Latitude')
                if i == len(axes) - 1:  # 最后一个子图
                    axes[i].invert_yaxis()
                axes[i].set_ylabel('Pressure (hPa)')
            else:
                # 如果当天有索引但所有值都是 NaN
                axes[i].text(0.5, 0.5, f'No valid data for Day {day}', 
                             ha='center', va='center', transform=axes[i].transAxes)
                
        except (IndexError, KeyError):
            # 如果当天在数据集中根本不存在
            axes[i].text(0.5, 0.5, f'No data for Day {day}', 
                         ha='center', va='center', transform=axes[i].transAxes)
    
    plt.tight_layout() #rect=[0, 0, 1, 0.96]
    plt.savefig(os.path.join(plot_dir, f'{variable_name.replace(" ", "_")}_timeline_{year}_day{start_day}-{end_day}.png'), dpi=400)
    plt.close(fig)
    print(f"✅ Saved timeline plot for {variable_name}.")

def plot_zonal_mean_anomaly_timeline(variable_name, start_day, end_day, step_days, plot_dir, OUTPUT_DIR,
                                   baseline_year=2021, target_year=2022,plot_args=None):
    """
    绘制变量随时间变化的global zonal mean异常值
    
    Parameters:
    -----------
    variable_name : str
        变量名称，如 'ML-Predicted OH'
    start_day : int
        起始天数
    end_day : int
        结束天数
    step_days : int
        步长（天）
    plot_dir : str
        图片保存目录
    baseline_year : int
        基准年份（用于计算异常值）
    target_year : int
        目标年份（要绘制异常值的年份）
    plot_args : dict, optional
        其他绘图参数，如图形大小等
    """
    if plot_args is None:
        plot_args = {'fig_h': 10, 'fig_w': 10, 'cmap':'RdBu_r'}
    
    fig_h = plot_args.get('fig_h', 3)
    fig_w = plot_args.get('fig_w', 10)
    fig_cmap = plot_args.get('cmap', 'RdBu_r')



    print(f"\n--- Generating Zonal Mean Anomaly Timeline for {variable_name} ---")
    print(f"Baseline year: {baseline_year}, Target year: {target_year}")
    
    # 确定数据文件名
    if variable_name == 'ML-Predicted OH':
        baseline_fname =f'predicted_MLS_OH_density_gridded_{baseline_year}.nc'#f'calculated_SSA_OH_density_gridded_{baseline_year}.nc' #
        target_fname = f'predicted_MLS_OH_density_gridded_{target_year}.nc'
    elif variable_name == 'MLS OH Obs':
        baseline_fname = f'true_MLS_OH_density_gridded_{baseline_year}.nc'
        target_fname = f'true_MLS_OH_density_gridded_{target_year}.nc'
    elif variable_name == 'Calculated SSA-OH':
        baseline_fname = f'calculated_SSA_OH_density_gridded_{baseline_year}.nc'
        target_fname = f'calculated_SSA_OH_density_gridded_{target_year}.nc'
    elif variable_name == 'Predicted SSA-OH':
        baseline_fname = f'predicted_SSA_OH_density_gridded_{baseline_year}.nc'
        target_fname = f'predicted_SSA_OH_density_gridded_{target_year}.nc'
    elif variable_name == 'TOMCAT OH':
        baseline_fname = f'TOMCAT_OH_density_gridded_{baseline_year}.nc'
        target_fname = f'TOMCAT_OH_density_gridded_{target_year}.nc'
    else:
        print(f"Unknown variable name: {variable_name}")
        return
    
    # 加载基准年和目标年数据
    try:
        baseline_ds = xr.open_dataset(os.path.join(OUTPUT_DIR, baseline_fname))
        target_ds = xr.open_dataset(os.path.join(OUTPUT_DIR, target_fname))
        var_name1 = list(baseline_ds.data_vars)[0]
        var_name2 = list(target_ds.data_vars)[0]

    except FileNotFoundError as e:
        print(f"Could not load data: {e}")
        return
    
    # 设置压力层范围
    plev_min = 0.9
    plev_max = 33
    
    # 生成时间序列
    days = range(start_day, end_day + 1, step_days)
    n_days = len(days)
    
    # 创建子图
    fig, axes = plt.subplots(nrows=n_days, ncols=1, figsize=(fig_w, fig_h*n_days), sharex=True, sharey=True)
    if n_days == 1:
        axes = [axes]
    
    # fig.suptitle(f'{variable_name} Global Zonal Mean Anomaly Timeline\n({target_year} - {baseline_year})', fontsize=16)
    
    # 收集所有异常值数据用于确定颜色范围
    all_anomalies = []
    for day in days:
        try:
            # 获取基准年数据
            baseline_daily = baseline_ds[var_name1].sel(doy=day, method="nearest")
            baseline_daily = baseline_daily.sel(pressure=slice(plev_min, plev_max))
            baseline_zonal = baseline_daily.mean(dim='lon', skipna=True)
            
            # 获取目标年数据
            target_daily = target_ds[var_name2].sel(doy=day, method="nearest")
            target_daily = target_daily.sel(pressure=slice(plev_min, plev_max))
            target_zonal = target_daily.mean(dim='lon', skipna=True)
            
            # 计算异常值
            anomaly = (target_zonal - baseline_zonal)/ baseline_zonal*100  # 计算百分比异常
            all_anomalies.append(anomaly)
        except (IndexError, KeyError):
            continue
    
    if not all_anomalies:
        print("No anomaly data available to plot.")
        return
    
    # 确定颜色范围
    # max_anomaly = max(d.max() for d in all_anomalies)
    # min_anomaly = min(d.min() for d in all_anomalies)
    # abs_max = max(abs(max_anomaly), abs(min_anomaly))
    abs_max = 150  # 设置异常值的最大绝对值范围为100%

    # 绘制每个时间点的异常值图
    for i, day in enumerate(days):
        try:
            # 获取基准年数据
            baseline_daily = baseline_ds[var_name1].sel(doy=day, method="nearest")
            baseline_daily = baseline_daily.sel(pressure=slice(plev_min, plev_max))
            baseline_zonal = baseline_daily.mean(dim='lon', skipna=True)
            
            # 获取目标年数据
            target_daily = target_ds[var_name2].sel(doy=day, method="nearest")
            target_daily = target_daily.sel(pressure=slice(plev_min, plev_max))
            target_zonal = target_daily.mean(dim='lon', skipna=True)
            
            # 计算异常值
            anomaly = (target_zonal - baseline_zonal)/ baseline_zonal*100  # 计算百分比异常
            
            # 绘制异常值
            anomaly.plot.pcolormesh(ax=axes[i], x='lat', y='pressure', 
                                  cmap=fig_cmap, vmin=-abs_max, vmax=abs_max,
                                  cbar_kwargs={'label': 'Anomaly (molec cm⁻³)'})
            # --- 这是关键修改 ---
            # 2. 将 'day' (一年中的第几天) 和 'year' 转换为日期对象
            date_obj = datetime.strptime(f'{target_year} {day}', '%Y %j')
            
            # 3. 格式化日期字符串 (例如: "January 10")
            #    使用 .day 而不是 %d 来避免日期的前导零 (例如 "January 1" 而不是 "January 01")
            date_str = f"{date_obj.strftime('%B')} {date_obj.day}"
            
            # 4. 设置新的标题
            axes[i].set_title(f'Day {day} ({date_str})')

            axes[i].set_yscale('log')
            axes[i].invert_yaxis()
            if i == len(axes) - 1:  # 最后一个子图
                axes[i].set_xlabel('Latitude')
            axes[i].set_ylabel('Pressure (hPa)')
        except (IndexError, KeyError):
            axes[i].text(0.5, 0.5, f'No data for Day {day}', 
                        ha='center', va='center', transform=axes[i].transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{variable_name.replace(" ", "_")}_anomaly_timeline_{target_year}-{baseline_year}_day{start_day}-{end_day}.png'), dpi=300)
    plt.close(fig)
    print(f"✅ Saved anomaly timeline plot for {variable_name}.")



def plot_temporal_trends(years, p_levels_to_plot, OUTPUT_DIR, plot_dir):
    print("\n--- Generating Temporal Trend Plots ---")
    all_monthly_means = []
    data_vars = {
        'MLS OH Obs': 'true_MLS_OH_density',
        'ML-Predicted OH': 'predicted_MLS_OH_density',
        'Calculated SSA-OH': 'calculated_SSA_OH_density',
    }

    for year in tqdm(years, desc="Processing years for temporal plots"):
        for p_level in p_levels_to_plot:
            for var_title, var_name in data_vars.items():
                try:
                    ds = xr.open_dataset(os.path.join(OUTPUT_DIR, f'{var_name}_gridded_{year}.nc'))
                    mean_data = ds[var_name].sel(pressure=p_level, method='nearest').mean(dim=['lat', 'lon']).to_dataframe(name='val').reset_index()
                    mean_data['Variable'] = var_title
                    mean_data['Pressure'] = p_level
                    mean_data['Year'] = year
                    mean_data['Date'] = pd.to_datetime(mean_data['Year'].astype(str) + '-' + mean_data['doy'].astype(str), format='%Y-%j')
                    all_monthly_means.append(mean_data)
                except (FileNotFoundError, IndexError):
                    continue

    if not all_monthly_means:
        print("No data processed for temporal trends."); return

    final_df = pd.concat(all_monthly_means).reset_index(drop=True)

    def reindex_group(df):
        df = df.set_index('Date')
        if df.empty:
            return None
        df_resampled = df.resample('M').mean()
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='M')
        df_reindexed = df_resampled.reindex(full_range)
        df_reindexed.index.name = 'Date'
        return df_reindexed

    final_df_monthly = final_df.groupby(['Variable', 'Pressure']).apply(reindex_group).reset_index()

    g = sns.FacetGrid(final_df_monthly, col="Pressure", col_wrap=1, hue="Variable",
                      height=4, aspect=5, sharey=False, col_order=sorted(p_levels_to_plot))

    g.map(plt.plot, "Date", "val", marker='o', linestyle='-', markersize=2, alpha=0.8)

    g.add_legend().set_axis_labels("Year", "Monthly Mean OH Density (molec/cm³)")
    g.set_titles("Pressure Level: {col_name} hPa").fig.suptitle('Global Monthly Mean OH Concentration Time Series', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'temporal_trends_matplotlib_fix.png'), dpi=300)
    plt.close('all')
    print("✅ 已使用Matplotlib直接绘图并保存。")

        # --- 2. 新增：绘制所有压力层总和的图 ---
    print("\n正在绘制所有压力层总和的趋势图...")
    
    # 按日期和变量对月度数据进行分组，并对'val'（OH浓度）求和
    # 这会得到每个时间点上，所有压力层的OH浓度总和
    total_oh_df = final_df_monthly.groupby(['Date', 'Variable'])['val'].sum(min_count=1).reset_index()

    plt.figure(figsize=(20, 7))
    for var in total_oh_df['Variable'].unique():
        df_var = total_oh_df[total_oh_df['Variable'] == var]
        plt.plot(df_var['Date'], df_var['val'], marker='o', linestyle='-', markersize=4, alpha=0.8, label=var)

    plt.title('Global Monthly Mean OH Concentration Time Series', fontsize=16, y=1.02)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Monthly Mean OH Density (molec/cm³)", fontsize=12) # 单位已更改，因为是柱浓度
    plt.legend(title='Type')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # 保存总和图
    output_path_total_sum = os.path.join(plot_dir, 'temporal_trends_total_sum.png')
    plt.savefig(output_path_total_sum, dpi=300)
    plt.close('all')
    print(f"✅ 所有压力层总和的图已保存至: {output_path_total_sum}")


def plot_zonal_mean_comparison_sci(year, day_of_year, plot_dir, output_dir):
    """
    以科研出版风格生成纬向平均OH浓度对比图 (V2 - 紧凑布局)。

    主要改进:
    - 将评估指标格式化为紧凑的多列布局。
    - 调整图表和子图间距，减少多余空白，使整体更紧凑。
    - 使用等宽字体确保指标列对齐。
    """
    print(f"\n--- 正在为 {year} 年, 第 {day_of_year} 天生成科研风格对比图 (紧凑布局) ---")
     # --- 字体加载部分保持不变 ---
    font_dir = 'fonts'
    if os.path.exists(font_dir):
        print(f"正在从 '{font_dir}' 加载本地字体...")
        for font_file in fm.findSystemFonts(fontpaths=[font_dir]):
            fm.fontManager.addfont(font_file)
        print("本地字体加载完成。")
    else:
        print(f"警告: 本地字体文件夹 '{font_dir}' 不存在，将使用系统默认字体。")

    # --- 1. 数据加载 --- (此部分不变)
    data_sources = {
        'ML-Predicted OH': f'predicted_MLS_OH_density_gridded_{year}.nc',
        'Calculated SSA-OH': f'calculated_SSA_OH_density_gridded_{year}.nc',
        'Predicted SSA-OH': f'predicted_SSA_OH_density_gridded_{year}.nc',
    }
    loaded_data = {}
    for title, fname in data_sources.items():
        filepath = os.path.join(output_dir, fname)
        data_array = _load_and_process_data(filepath, day_of_year)
        if data_array is not None:
            loaded_data[title] = data_array
    # 从constant-var 路径加载 True MLS OH 和 TOMCAT OH
    true_mls_path = os.path.join(Constant_var_path, f'true_MLS_OH_density_gridded_{year}.nc')
    print(f"Loading True MLS OH data from: {true_mls_path}")
    tomcat_path = os.path.join(Constant_var_path, f'TOMCAT_OH_density_gridded_{year}.nc')
    true_mls_data = _load_and_process_data(true_mls_path, day_of_year)
    if true_mls_data is not None:
        loaded_data['True MLS OH'] = true_mls_data
    tomcat_data = _load_and_process_data(tomcat_path, day_of_year)
    if tomcat_data is not None:
        loaded_data['TOMCAT OH'] = tomcat_data


    if 'True MLS OH' not in loaded_data:
        print("错误: 必须有 'True MLS OH' 数据才能进行比较。已跳过绘图。")
        return

    # --- 2. 指标计算 (修改为分列逻辑) ---
    comparisons = {
        "Pred. MLS vs True": ('ML-Predicted OH', 'True MLS OH'),
        "Calc. SSA vs True": ('Calculated SSA-OH', 'True MLS OH'),
        "Pred. SSA vs True": ('Predicted SSA-OH', 'True MLS OH'),
        "TOMCAT vs True": ('TOMCAT OH', 'True MLS OH')
    }
    
    comp_items = list(comparisons.items())
    mid_point = (len(comp_items) + 1) // 2  # 计算分割点
    col1_items, col2_items = comp_items[:mid_point], comp_items[mid_point:]
    
    col1_lines, col2_lines = [], []
    true_data = loaded_data['True MLS OH'].values

    # 处理第一列
    for label, (pred_key, true_key) in col1_items:
        line = f"  - {label}:"
        if pred_key in loaded_data:
            metrics = _calculate_metrics(true_data, loaded_data[pred_key].values)
            if metrics:
                line += f" RMSE={metrics['rmse']:.2e}, $R^2$={metrics['r2']:.3f}, SSIM={metrics['ssim']:.3f}"
            else: line += " Insufficient data"
        else: line += " Missing data"
        col1_lines.append(line)

    # 处理第二列
    for label, (pred_key, true_key) in col2_items:
        line = f"  - {label}:"
        if pred_key in loaded_data:
            metrics = _calculate_metrics(true_data, loaded_data[pred_key].values)
            if metrics:
                line += f" RMSE={metrics['rmse']:.2e}, $R^2$={metrics['r2']:.3f}, SSIM={metrics['ssim']:.3f}"
            else: line += " Insufficient data"
        else: line += " Missing data"
        col2_lines.append(line)

    # 格式化为单字符串，使用等宽字体对齐
    max_len_col1 = max(len(s) for s in col1_lines) if col1_lines else 0
    formatted_lines = []
    for i in range(len(col1_lines)):
        line1 = col1_lines[i]
        line2 = col2_lines[i] if i < len(col2_lines) else ""
        # 使用 f-string 的对齐功能，并增加额外间距
        formatted_lines.append(f"{line1:<{max_len_col1}}    {line2}")
    metrics_text_aligned = "\n".join(formatted_lines)
    
    # --- 3. 绘图风格设置 --- (此部分不变)
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
        'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'axes.titlesize': 12, 'figure.titlesize': 12, 'axes.linewidth': 1.2,
    })

    # --- 4. 绘图 --- (此部分不变)
    plot_order = ['True MLS OH', 'ML-Predicted OH', 'Calculated SSA-OH', 'TOMCAT OH']
    plot_data = {k: loaded_data[k] for k in plot_order if k in loaded_data}
    num_plots = len(plot_data)
    if num_plots == 0: return

    # 调整画布高度以实现紧凑布局
    fig, axes = plt.subplots(
        nrows=1, ncols=num_plots, 
        figsize=(5 * num_plots, 6.5), # 减小了高度
        sharex=True, sharey=True,
        gridspec_kw={'bottom': 0.1}  # 增加底部空间以容纳文本
    )
    if num_plots == 1: axes = [axes]
    
    vmax = loaded_data['True MLS OH'].max()
    cbar_label = r'OH Density (10$^{6}$ cm$^{-3}$)'

    for i, (title, data) in enumerate(plot_data.items()):
        ax = axes[i]
        pcm = data.plot.pcolormesh(
            ax=ax, x='lat', y='pressure', cmap='turbo',
            vmin=0, vmax=25, add_colorbar=False
        )
        ax.set_title(title)
        ax.set_xlabel(''); ax.set_ylabel('')
        # 添加指标到对比子图下方（vs True）
        if title != 'True MLS OH':
            metrics = _calculate_metrics(loaded_data['True MLS OH'].values, data.values)
            if metrics:
                metric_str = f"RMSE: {metrics['rmse']:.2f}  R²: {metrics['r2']:.3f}  SSIM: {metrics['ssim']:.3f}"
                ax.text(0.5, -0.14, metric_str, ha='center', va='top', transform=ax.transAxes, fontsize=11)
    # --- 5. 布局与美化 (修改部分) ---
    fig.supxlabel('Latitude (°N)', fontsize=14, y=0.02)
    fig.supylabel('Pressure (hPa)', fontsize=14, x=0.06)
    
    axes[0].set_yscale('log')
    # axes[0].invert_yaxis()
    # 修正y轴范围：使用True MLS OH的精确压力范围，避免自动扩展
    true_data = loaded_data['True MLS OH']
    max_p = true_data.pressure.max().values
    min_p = true_data.pressure.min().values
    axes[0].set_ylim(max_p, min_p)  # 设置为max到min以反转轴，无需invert_yaxis()

    # 调整布局以适应更紧凑的文本框
    fig.subplots_adjust(left=0.09, right=0.86, top=0.85, bottom=0.28)
    
    cbar_ax = fig.add_axes([0.88, 0.1, 0.015, 0.75])
    fig.colorbar(pcm, cax=cbar_ax, label=cbar_label)
    # adjust colorbar limits
    # pcm.set_clim(0, 30)

    # --- MODIFICATION: 添加日期文本 ---
    data_date_obj = datetime.strptime(f'{year}-{day_of_year}', '%Y-%j')
    formatted_date = data_date_obj.strftime('%d %b %Y')    
    fig.text(
        0.9, 0.9,
        f'Date: {formatted_date}', 
        ha='right', va='bottom',
        fontsize=11, color='grey', alpha=0.5
    )
    # 添加指标文本框 (新布局)
    # a. 添加标题
    # fig.text(0.5, 0.18, "Evaluation Metrics", ha="center", va="bottom", fontsize=14, weight='bold')
    
    # b. 添加对齐后的指标
    # fig.text(0.5, 0, metrics_text_aligned,
    #          ha="center", va="top",
    #          fontsize=10,  # 使用稍小的字号
    #          fontfamily='monospace',  # 使用等宽字体确保对齐
    #          )

    # fig.suptitle(f'Zonal Mean OH Comparison - {year}, Day {day_of_year}', fontsize=18)
    
    plot_path = os.path.join(plot_dir, f'zonal_mean_comparison_{year}_day{day_of_year}_sci_compact.png')
    plt.savefig(plot_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 已保存科研风格对比图 (紧凑布局): {plot_path}")



def plot_comprehensive_error_analysis(years, OUTPUT_DIR, plot_dir):
    """
    生成一个紧凑的"四合一"误差分析图，现在包括 TOMCAT OH 的分析。
    """
    # --- 字体加载部分保持不变 ---
    font_dir = 'fonts'
    if os.path.exists(font_dir):
        print(f"正在从 '{font_dir}' 加载本地字体...")
        for font_file in fm.findSystemFonts(fontpaths=[font_dir]):
            fm.fontManager.addfont(font_file)
        print("本地字体加载完成。")
    else:
        print(f"警告: 本地字体文件夹 '{font_dir}' 不存在，将使用系统默认字体。")

    print(f"\n--- 开始为 {years} 年进行综合误差分析 (包含TOMCAT OH) ---")
    
    daily_metrics = []
    pressure_level_metrics = []

    for year in years:
        try:
            # --- 1. 数据加载 (新增TOMCAT) ---
            ds_true = xr.open_dataset(os.path.join(Constant_var_path, f'true_MLS_OH_density_gridded_{year}.nc'))
            ds_pred = xr.open_dataset(os.path.join(OUTPUT_DIR, f'predicted_MLS_OH_density_gridded_{year}.nc'))
            ds_calc = xr.open_dataset(os.path.join(OUTPUT_DIR, f'calculated_SSA_OH_density_gridded_{year}.nc'))
            ds_tomcat = xr.open_dataset(os.path.join(Constant_var_path, f'TOMCAT_OH_density_gridded_{year}.nc')) # <-- 新增
            
            true_var, pred_var, calc_var, tomcat_var = list(ds_true.data_vars)[0], list(ds_pred.data_vars)[0], list(ds_calc.data_vars)[0], list(ds_tomcat.data_vars)[0]
            
            # --- 更新共同日期的计算 ---
            common_days = np.intersect1d(ds_true.doy.values, ds_pred.doy.values)
            common_days = np.intersect1d(common_days, ds_calc.doy.values)
            common_days = np.intersect1d(common_days, ds_tomcat.doy.values) # <-- 新增
            pressure_levels = ds_true.pressure.values

            print(f"正在处理 {year} 年的 {len(common_days)} 天数据...")
            for doy in tqdm(common_days, desc=f"分析 {year}", leave=False):
                date = pd.to_datetime(f'{year}-{doy}', format='%Y-%j')
                
                true_zonal = ds_true[true_var].sel(doy=doy).mean(dim='lon', skipna=True)
                pred_zonal = ds_pred[pred_var].sel(doy=doy).mean(dim='lon', skipna=True)
                calc_zonal = ds_calc[calc_var].sel(doy=doy).mean(dim='lon', skipna=True)
                tomcat_zonal = ds_tomcat[tomcat_var].sel(doy=doy).mean(dim='lon', skipna=True) # <-- 新增

                # --- 2. 计算每日总体指标 (新增TOMCAT) ---
                metrics_pred = _calculate_metrics_2(true_zonal.values, pred_zonal.values)
                metrics_calc = _calculate_metrics_2(true_zonal.values, calc_zonal.values)
                metrics_tomcat = _calculate_metrics_2(true_zonal.values, tomcat_zonal.values) # <-- 新增

                if not np.isnan(metrics_pred['rmse']):
                    daily_metrics.append({'date': date, 'month': date.month, **metrics_pred, 'type': 'ML-Predicted OH'})
                if not np.isnan(metrics_calc['rmse']):
                    daily_metrics.append({'date': date, 'month': date.month, **metrics_calc, 'type': 'Calculated SSA-OH'})
                if not np.isnan(metrics_tomcat['rmse']):
                    daily_metrics.append({'date': date, 'month': date.month, **metrics_tomcat, 'type': 'TOMCAT OH'}) # <-- 新增
                # --- 3. 计算每个压力层上的指标 (新增TOMCAT) ---
                for p_level in pressure_levels:
                    true_at_p = true_zonal.sel(pressure=p_level).values
                    pred_at_p = pred_zonal.sel(pressure=p_level).values
                    calc_at_p = calc_zonal.sel(pressure=p_level).values
                    tomcat_at_p = tomcat_zonal.sel(pressure=p_level).values # <-- 新增

                    metrics_p_pred = _calculate_metrics_2(true_at_p, pred_at_p)
                    metrics_p_calc = _calculate_metrics_2(true_at_p, calc_at_p)
                    metrics_p_tomcat = _calculate_metrics_2(true_at_p, tomcat_at_p) # <-- 新增

                    if not np.isnan(metrics_p_pred['rmse']):
                        pressure_level_metrics.append({'pressure': p_level, 'rmse': metrics_p_pred['rmse'], 'type': 'ML-Predicted OH'})
                    if not np.isnan(metrics_p_calc['rmse']):
                        pressure_level_metrics.append({'pressure': p_level, 'rmse': metrics_p_calc['rmse'], 'type': 'Calculated SSA-OH'})
                    if not np.isnan(metrics_p_tomcat['rmse']):
                        pressure_level_metrics.append({'pressure': p_level, 'rmse': metrics_p_tomcat['rmse'], 'type': 'TOMCAT OH'}) # <-- 新增

        except FileNotFoundError as e:
            print(f"警告: 找不到 {year} 年所需的数据文件，已跳过。缺失文件: {e.filename}")
            continue
    
    if not daily_metrics:
        print("未能处理足够的数据以生成图表。")
        return

    df_daily = pd.DataFrame(daily_metrics)
    df_pressure = pd.DataFrame(pressure_level_metrics)

    font_settings = {
        'title': 18,        # Title font size for each subplot
        'label': 14,        # Axis label font size (x and y)
        'ticks': 12,        # Axis tick label font size
        'legend_title': 12, # Legend title font size
        'legend_text': 10,   # Legend item font size
        'offset_text': 12   # <-- New: Font size for axis multipliers (e.g., 1e6)

    }

    # Update global font settings
    plt.rcParams.update({
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': font_settings['ticks'] # Set a default font size
    })
    # --- 4. 绘图 (更新调色板) ---
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.sans-serif': ['Arial', 'Helvetica']})
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), gridspec_kw={'height_ratios': [2, 2]})

    # --- Palette including TOMCAT ---
    palette = {
        'ML-Predicted OH': 'royalblue',
        'Calculated SSA-OH': 'darkorange',
        'TOMCAT OH': 'forestgreen'
    }

    # --- Plot 1: RMSE Distribution ---
    sns.histplot(data=df_daily, x='rmse', hue='type', bins=200, kde=True, ax=axes[0, 0], palette=palette, stat='probability')
    axes[0, 0].set_xlim(0.9, 4)
    axes[0, 0].set_title('a) Daily RMSE Distribution', fontsize=font_settings['title'])
    axes[0, 0].set_xlabel(r'RMSE (molec./10$^{6}$ cm$^{-3}$)', fontsize=font_settings['label'])
    axes[0, 0].set_ylabel('Probability', fontsize=font_settings['label'])
    axes[0, 0].tick_params(axis='both', labelsize=font_settings['ticks']) # Control tick label size
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    sns.move_legend(axes[0, 0], "upper right", title='Comparison Type',
                    title_fontsize=font_settings['legend_title'], fontsize=font_settings['legend_text'])

    # --- Plot 2: R² Distribution ---
    sns.histplot(data=df_daily, x='r2', hue='type', bins=200, kde=True, ax=axes[0, 1], palette=palette,stat='probability')
    axes[0, 1].set_xlim(0.8, 1)
    axes[0, 1].set_title(r'b) Daily $R^2$ Distribution', fontsize=font_settings['title'])
    axes[0, 1].set_xlabel(r'Coefficient of Determination ($R^2$)', fontsize=font_settings['label'])
    axes[0, 1].set_ylabel('Frequency', fontsize=font_settings['label'])
    axes[0, 1].tick_params(axis='both', labelsize=font_settings['ticks']) # Control tick label size
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    sns.move_legend(axes[0, 1], "upper left", title='Comparison Type',
                    title_fontsize=font_settings['legend_title'], fontsize=font_settings['legend_text'])

    boxplot_kwargs = {"boxprops": {'alpha': 0.7}}

    # --- Plot 3: Monthly RMSE Boxplot ---
    sns.boxplot(data=df_daily, x='month', y='rmse', hue='type', ax=axes[1, 0], palette=palette, showfliers=False, **boxplot_kwargs)
    axes[1, 0].set_ylim(0.9, 2.75)
    axes[1, 0].set_title('c) Monthly RMSE Statistics', fontsize=font_settings['title'])
    axes[1, 0].set_xlabel('Month', fontsize=font_settings['label'])
    axes[1, 0].set_ylabel(r'Daily RMSE (molec./10$^{6}$ cm$^{-3})$', fontsize=font_settings['label'])
    axes[1, 0].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    axes[1, 0].tick_params(axis='both', labelsize=font_settings['ticks']) # Control tick label size
    axes[1, 0].grid(True, axis='y', linestyle='--', alpha=0.6)
    sns.move_legend(axes[1, 0], "upper right", title='Comparison Type',
                    title_fontsize=font_settings['legend_title'], fontsize=font_settings['legend_text'])
    import matplotlib.ticker as mticker
    # --- Plot 4: Pressure Level RMSE Boxplot (Horizontal) ---
    sns.boxplot(data=df_pressure, y='pressure', x='rmse', hue='type', ax=axes[1, 1], palette=palette, orient='h', showfliers=False, **boxplot_kwargs)
    axes[1, 1].set_title('d) RMSE by Pressure Level', fontsize=font_settings['title'])
    axes[1, 1].set_ylabel('Pressure (hPa)', fontsize=font_settings['label'])
    axes[1, 1].set_xlabel(r'RMSE (molec./10$^{6}$ cm$^{-3}$)', fontsize=font_settings['label'])
    axes[1, 1].tick_params(axis='both', labelsize=font_settings['ticks']) # Control tick label size

    # 1. 获取当前Y轴的刻度标签 (它们是字符串)
    current_labels = [item.get_text() for item in axes[1, 1].get_yticklabels()]

    # 2. 将它们转换为浮点数，格式化为两位小数，然后存为新列表
    #    使用 try-except 来避免标签为空或非数字时出错
    new_labels = []
    for label in current_labels:
        try:
            # 将 "850" -> 850.0 -> "850.00"
            new_labels.append(f"{float(label):.2f}")
        except ValueError:
            # 如果标签不是数字 (例如空字符串)，保持原样
            new_labels.append(label)

    # 3. 将新格式化的标签设置回去
    axes[1, 1].set_yticklabels(new_labels)

    # axes[1, 1].invert_yaxis()
    axes[1, 1].grid(True, axis='x', linestyle='--', alpha=0.6)
    sns.move_legend(axes[1, 1], "upper right", title='Comparison Type',
                    title_fontsize=font_settings['legend_title'], fontsize=font_settings['legend_text'])
    
    axes[0, 0].xaxis.get_offset_text().set_size(font_settings['offset_text']) # <-- New: Adjust size of '1e6'
    axes[0, 1].xaxis.get_offset_text().set_size(font_settings['offset_text']) # <-- New: Adjust size of '1e6'
    axes[1, 0].yaxis.get_offset_text().set_size(font_settings['offset_text']) # <-- New: Adjust size of '1e6'
    axes[1, 1].xaxis.get_offset_text().set_size(font_settings['offset_text']) # <-- New: Adjust size of '1e6'   

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plot_path = os.path.join(plot_dir, f'comprehensive_error_analysis_{years[0]}-{years[-1]}.png')
    plt.savefig(plot_path, dpi=400)
    plt.close(fig)
    print(f"✅ 已保存包含TOMCAT的综合误差分析图: {plot_path}")


def plot_hunga_tonga_analysis(plot_year, baseline_year, doy, plot_dir, OUTPUT_DIR, pressure_level=10.0):
    """
    绘制Hunga-Tonga火山爆发影响分析图 (V11 - 调整为3列布局，移除Raw Data图)。
    """
    print(f"\n--- Generating Hunga-Tonga Eruption Analysis for doy {doy} ---")

    def _load_and_process_day(variable_name, year, day_of_year, output_dir):
        # ... (此内部函数保持不变) ...
        if variable_name == 'ML-Predicted OH':
            fname = f'predicted_MLS_OH_density_gridded_{year}.nc'
        elif variable_name.upper() == 'H2O':
            fname = f'H2O_gridded_{year}.nc'
        else: return None
        filepath = os.path.join(output_dir, fname)
        try:
            ds = xr.open_dataset(filepath)
            var_name = list(ds.data_vars)[0]
            daily_data = ds[var_name].sel(doy=day_of_year, method="nearest")
            return daily_data
        except (FileNotFoundError, IndexError, KeyError):
            print(f"Info: Could not load or process file: {fname}")
            return None

    #
    title_fontsize = 10
    # 1. 加载和计算数据
    h2o_plot_data = _load_and_process_day('H2O', plot_year, doy, Constant_var_path)*1e6 # 转换为 ppmv
    h2o_base_data = _load_and_process_day('H2O', baseline_year, doy, Constant_var_path)*1e6 # 转换为 ppmv
    oh_plot_data = _load_and_process_day('ML-Predicted OH', plot_year, doy, OUTPUT_DIR)
    oh_base_data = _load_and_process_day('ML-Predicted OH', baseline_year, doy, OUTPUT_DIR)
    if any(data is None for data in [h2o_plot_data, h2o_base_data, oh_plot_data, oh_base_data]):
        print("Error: Missing necessary data. Aborting plot."); return
    h2o_zonal_plot = h2o_plot_data.mean(dim='lon', skipna=True)
    h2o_zonal_anomaly = (h2o_zonal_plot - h2o_base_data.mean(dim='lon', skipna=True)) / h2o_base_data.mean(dim='lon', skipna=True) * 100
    oh_zonal_plot = oh_plot_data.mean(dim='lon', skipna=True)
    oh_zonal_anomaly = (oh_zonal_plot - oh_base_data.mean(dim='lon', skipna=True)) / oh_base_data.mean(dim='lon', skipna=True) * 100
    
    # --- MODIFICATION: 只需为插值图准备数据 ---
    oh_map_plot_raw = oh_plot_data.sel(pressure=pressure_level, method='nearest')
    h2o_map_plot_raw = h2o_plot_data.sel(pressure=pressure_level, method='nearest')

    # 调用新的加权网格化函数
    SEARCH_FACTOR = 3.0
    lon_centers_oh, lat_centers_oh, oh_map_gridded = perform_weighted_gridding(
        data_array=oh_map_plot_raw,
        lat_spacing=2.0, lon_spacing=5.0,
        lat_half_width=1.5, lon_half_width=8.0,
        search_radius_factor=SEARCH_FACTOR
    )
    lon_centers_h2o, lat_centers_h2o, h2o_map_gridded = perform_weighted_gridding(
        data_array=h2o_map_plot_raw,
        lat_spacing=2.0, lon_spacing=5.0,
        lat_half_width=1.5, lon_half_width=8.0,
        search_radius_factor=SEARCH_FACTOR
    )

    if oh_map_gridded is None or h2o_map_gridded is None:
        print("Error: Not enough data for weighted gridding. Aborting plot.")
        return

    # 3. 绘图
    plt.style.use('seaborn-v0_8-paper')
    # --- MODIFICATION: 调整画布尺寸以适应3列布局 ---
    fig = plt.figure(figsize=(20, 9))

    # --- MODIFICATION: 调整GridSpec以适应3列布局 ---
    # 外部网格：左侧(2列 zonal) vs 右侧(1列 map)
    gs_outer = gridspec.GridSpec(1, 2, width_ratios=[2, 2], wspace=0.15)

    # 左侧内部网格：2x2 用于 zonal plots
    gs_left = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_outer[0],
                                               width_ratios=[1, 1], wspace=0.2, hspace=0.15)

    # 右侧内部网格：2x1 用于 interpolated maps
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[1],
                                                hspace=0.15) # 仅需行间距

    # --- MODIFICATION: 创建3列的axes ---
    axes = np.empty((2, 3), dtype=object)

    # 创建左侧的子图 (第1列和第2列)
    axes[0, 0] = fig.add_subplot(gs_left[0, 0])
    axes[1, 0] = fig.add_subplot(gs_left[1, 0])
    axes[0, 1] = fig.add_subplot(gs_left[0, 1])
    axes[1, 1] = fig.add_subplot(gs_left[1, 1])

    # 创建右侧的子图 (第3列)
    projection = ccrs.PlateCarree(central_longitude=180) if CARTOPY_AVAILABLE else None
    axes[0, 2] = fig.add_subplot(gs_right[0, 0], projection=projection)
    axes[1, 2] = fig.add_subplot(gs_right[1, 0], projection=projection)
    
    # --- Col 1 & 2: Zonal Mean Plots (代码基本不变) ---
    zonal_plot_bounds = [0, 0.05, 1, 0.9]  
    for i in range(2):
        for j in range(2):
            parent_ax = axes[i, j]
            parent_ax.axis('off')
            ax = parent_ax.inset_axes(zonal_plot_bounds)
            divider = make_axes_locatable(ax)
            cax_for_colorbar = divider.append_axes("right", size="5%", pad=0.1)
            
            if j == 0 and i == 0:
                data = h2o_zonal_plot
                mappable = ax.pcolormesh(data.lat, data.pressure, data, cmap='turbo')
                cb = fig.colorbar(mappable, cax=cax_for_colorbar)
                cb.ax.set_title(r'H2O ppmv', fontsize=8)
                ax.set_title(f'a) H2O Zonal Mean', fontsize=title_fontsize)
            elif j == 0 and i == 1:
                data = h2o_zonal_anomaly
                mappable = ax.pcolormesh(data.lat, data.pressure, data, cmap='RdBu_r', vmin=-100, vmax=100)
                cb = fig.colorbar(mappable, cax=cax_for_colorbar)
                cb.ax.set_title(r'    Anomaly (%)', fontsize=8)
                ax.set_title(f'b) H2O Zonal Mean Relative Anomaly)', fontsize=title_fontsize)
            elif j == 1 and i == 0:
                data = oh_zonal_plot
                mappable = ax.pcolormesh(data.lat, data.pressure, data, cmap='turbo')
                cb = fig.colorbar(mappable, cax=cax_for_colorbar)
                cb.ax.set_title(r'OH density (10$^{6}$ cm$^{-3}$)', fontsize=8)
                ax.set_title(f'c) OH Zonal Mean', fontsize=title_fontsize)
            elif j == 1 and i == 1:
                data = oh_zonal_anomaly
                mappable = ax.pcolormesh(data.lat, data.pressure, data, cmap='RdBu_r', vmin=-100, vmax=100)
                cb = fig.colorbar(mappable, cax=cax_for_colorbar)
                cb.ax.set_title(r'    Anomaly (%)', fontsize=8)
                ax.set_title(f'd) OH Zonal Mean Relative Anomaly)', fontsize=title_fontsize)  #\n({plot_year} vs {baseline_year}
            
            ax.set_yscale('log')
            # ax.invert_yaxis()
            
            max_p = data.pressure.max().values
            min_p = data.pressure.min().values
            ax.set_ylim(max_p, min_p)  # 设置为max到min以反转轴，无需invert_yaxis()
            ax.set_ylabel('Pressure (hPa)' if j == 0 else '')
            ax.set_xlabel('Latitude' if i == 1 else '')
            if j == 1: plt.setp(ax.get_yticklabels(), visible=False)

    # --- MODIFICATION: Col 3: Interpolated Global Maps ---
    ax_oh_map = axes[0, 2]
    ax_h2o_map = axes[1, 2]

    # # Interpolated OH Map
    # cf_oh = ax_oh_map.contourf(lon_grid, lat_grid, oh_map_plot_interp, levels=10, cmap='turbo', transform=ccrs.PlateCarree() if CARTOPY_AVAILABLE else None)
    # fig.colorbar(cf_oh, ax=ax_oh_map, label=r'OH density (cm$^{-3}$)', shrink=0.9)
    # ax_oh_map.set_title(f'Interpolated OH Global Map at {pressure_level} hPa', fontsize=title_fontsize)

    # # Interpolated H2O Map
    # cf_h2o = ax_h2o_map.contourf(lon_grid, lat_grid, h2o_map_plot_interp, levels=10, cmap='turbo', transform=ccrs.PlateCarree() if CARTOPY_AVAILABLE else None)
    # fig.colorbar(cf_h2o, ax=ax_h2o_map, label=r'H2O ppmv', shrink=0.9)
    # ax_h2o_map.set_title(f'Interpolated H2O Global Map at {pressure_level} hPa', fontsize=title_fontsize)

    # 注意：这里不再需要 levels 参数
# --- 修改点: 无需再使用 cyclic 变量，直接使用网格化函数的输出 ---
    cf_oh = ax_oh_map.contourf(lon_centers_oh, lat_centers_oh, oh_map_gridded, cmap='YlGnBu',alpha=0.8,
                               levels=20,
                               transform=ccrs.PlateCarree() if CARTOPY_AVAILABLE else None)
    fig.colorbar(cf_oh, ax=ax_oh_map, label=r'OH density (10$^{6}$ cm$^{-3}$)', shrink=0.9)
    ax_oh_map.set_title(f'e) Interpolated OH Global Map at {pressure_level} hPa', fontsize=title_fontsize)

    cf_h2o = ax_h2o_map.contourf(lon_centers_h2o, lat_centers_h2o, h2o_map_gridded, cmap='YlGnBu',alpha=0.8,
                                 levels=20,
                                 transform=ccrs.PlateCarree() if CARTOPY_AVAILABLE else None)
    fig.colorbar(cf_h2o, ax=ax_h2o_map, label=r'H2O ppmv', shrink=0.9)
    ax_h2o_map.set_title(f'f) Interpolated H2O Global Map at {pressure_level} hPa', fontsize=title_fontsize)



    # --- MODIFICATION: 美化地图循环 ---
    for i, ax in enumerate([ax_oh_map, ax_h2o_map]):
        if CARTOPY_AVAILABLE:
            ax.coastlines(color='grey')
            gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            # 只在最下面的地图上显示经度标签
            if i == 0: # Top map
                gl.bottom_labels = False
    
    # --- MODIFICATION: 添加日期文本 ---
    data_date_obj = datetime.strptime(f'{plot_year}-{doy}', '%Y-%j')
    formatted_date = data_date_obj.strftime('%d %b %Y')    
    fig.text(
        0.9, 0.9,
        f'Date: {formatted_date}', 
        ha='right', va='bottom',
        fontsize=11, color='grey', alpha=0.5
    )
    # 5. 保存图像
    plot_path = os.path.join(plot_dir, f'hunga_tonga_analysis_3col_{plot_year}_vs_{baseline_year}_day{doy}.png')
    plt.savefig(plot_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved final Hunga-Tonga analysis plot to: {plot_path}")


def plot_profile_comparison(year, doy, plot_dir, OUTPUT_DIR, num_profiles=10, random_seed=42, font_sizes=None):
    """
    绘制OH垂直剖面对比图。MLS观测的不确定度使用误差棒(error bars)形式展示，
    该不确定度根据压力值直接查找内置的精度与准确度表计算。
    """
    print(f"\n--- Generating OH Profile Comparison for {year}, Day {doy} ---")

    # --- 1. 设置与数据加载 ---
    default_font_sizes = {'title': 11, 'label': 12, 'tick': 10, 'legend': 12}
    if font_sizes is None: font_sizes = {}
    final_font_sizes = {**default_font_sizes, **font_sizes}

    data_sources = {
        'Predicted MLS-OH': f'predicted_MLS_OH_density_gridded_{year}.nc',
        'Predicted MLS-OH Uncertainty': f'predicted_MLS_OH_uncertainty_density_gridded_{year}.nc'
    }

    loaded_data = {}
    for key, fname in data_sources.items():
        filepath = os.path.join(OUTPUT_DIR, fname)
        try:
            ds = xr.open_dataset(filepath)
            var_name = list(ds.data_vars)[0]
            loaded_data[key] = ds[var_name].sel(doy=doy, method='nearest')
        except FileNotFoundError:
            print(f"Info: 未找到文件 '{fname}'，将跳过此数据源。")
    # 加载参考的 True MLS OH 数据
    true_mls_fname = f'true_MLS_OH_density_gridded_{year}.nc'
    true_mls_filepath = os.path.join(Constant_var_path, true_mls_fname)
    try:
        ds_true = xr.open_dataset(true_mls_filepath)
        var_name_true = list(ds_true.data_vars)[0]
        loaded_data['True MLS OH'] = ds_true[var_name_true].sel(doy=doy, method='nearest')
    except FileNotFoundError:
        print(f"Error: 未找到参考文件 '{true_mls_fname}'。绘图中止。")
        return

    # --- 2. 地理位置的均匀随机采样 ---
    # (此部分与之前版本相同)
    ref_data = loaded_data['True MLS OH']
    valid_points_stacked = ref_data.stack(points=('lat', 'lon')).dropna(dim='points', how='all')
    if len(valid_points_stacked.points) == 0:
        print(f"在第 {doy} 天未找到任何有效的 'True MLS OH' 数据点。")
        return
    num_to_sample = min(num_profiles, len(valid_points_stacked.points))
    rng = np.random.RandomState(seed=random_seed)
    num_bins = num_to_sample
    lat_bins = np.linspace(valid_points_stacked.lat.min().item(), valid_points_stacked.lat.max().item(), num_bins + 1)
    selected_points = []
    for i in range(num_bins):
        bin_points = valid_points_stacked.where(
            (valid_points_stacked.lat >= lat_bins[i]) & (valid_points_stacked.lat < lat_bins[i + 1]),
            drop=True
        )
        if len(bin_points.points) > 0:
            random_index = rng.choice(len(bin_points.points), size=1, replace=False)
            selected_points.append(bin_points.points.isel(points=random_index))
    selected_points = xr.concat(selected_points, dim='points').sortby('lat')

    # --- 3. 绘图 ---
    plt.style.use('seaborn-v0_8-ticks')
    ncols = 4
    nrows = (num_to_sample + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10), sharey=True)
    axes = axes.flatten() if num_to_sample > 1 else [axes]

    plot_styles = {
        'True MLS OH': {'color': 'red', 'label': 'MLS OH Obs and Uncertainty', 'zorder': 3},
        'Predicted MLS-OH': {'color': 'green', 'linestyle': '-', 'marker': '.', 'label': 'DRCAT-Predicted OH', 'zorder': 2}
    }

    for i, point in enumerate(selected_points):
        ax = axes[i]
        lat, lon = point.item()
        lat_str = f"{abs(lat):.1f}°N" if lat >= 0 else f"{abs(lat):.1f}°S"
        lon_str = f"{abs(lon):.1f}°E" if lon >= 0 else f"{abs(lon):.1f}°W"
        ax.set_title(f"{lat_str}, {lon_str}", fontsize=final_font_sizes['title'])

        # --- 绘图逻辑修改 ---
        # 1. 绘制 ML 预测值及其不确定度
        if 'Predicted MLS-OH' in loaded_data:
            style = plot_styles['Predicted MLS-OH']
            profile = loaded_data['Predicted MLS-OH'].sel(lat=lat, lon=lon, method='nearest')
            pressure = profile.pressure.values
            ax.plot(profile.values, pressure, **style)
            
            if 'Predicted MLS-OH Uncertainty' in loaded_data:
                unc_profile = loaded_data['Predicted MLS-OH Uncertainty'].sel(lat=lat, lon=lon, method='nearest')
                ax.fill_betweenx(pressure,
                                 profile.values - unc_profile.values,
                                 profile.values + unc_profile.values,
                                 color=style['color'], alpha=0.1, label='±Estimated Uncertainty')

        # 2. 绘制 MLS 观测值及其误差棒
        if 'True MLS OH' in loaded_data:
            style = plot_styles['True MLS OH']
            profile = loaded_data['True MLS OH'].sel(lat=lat, lon=lon, method='nearest')
            pressure = profile.pressure.values
            
            # 从内置表格计算总不确定度
            precision_profile = np.piecewise(pressure,
                [pressure < 10, (pressure >= 10) & (pressure < 14), pressure >= 14],
                [2.8, 4.7, 13.7]
            )
            accuracy_profile = np.piecewise(pressure,
                [pressure < 14, pressure >= 14],
                [1.0, 1.5]
            )
            total_uncertainty_profile = np.sqrt((precision_profile)**2 + (accuracy_profile)**2)
            
            # 使用 ax.errorbar 绘制
            ax.errorbar(profile.values, pressure, xerr=total_uncertainty_profile,
                        fmt='-',             # 仅绘制点标记
                        markersize=5,        # 标记大小
                        capsize=3,           # 误差棒端点帽的大小
                        elinewidth=0.2,        # 误差棒线宽
                        color=style['color'],
                        label=style['label'],
                        zorder=style['zorder'])

        ax.set_yscale('log')
        ax.grid(True, which='major', linestyle='--', alpha=0.7)
        ax.grid(True, which='minor', linestyle=':', alpha=0.4)
        ax.tick_params(axis='both', which='major', labelsize=final_font_sizes['tick'])
        if i >= len(axes) - ncols:
            ax.set_xlabel(r'OH Density (10$^{6}$ cm$^{-3}$)', fontsize=final_font_sizes['label'])

    # --- 4. 最终美化与保存 ---
    # (此部分与之前版本相同)
    for i in range(num_to_sample, len(axes)):
        axes[i].axis('off')

    if num_to_sample > 0:
        for i in range(nrows):
            ax = axes[i * ncols]
            if i * ncols < num_to_sample:
                ax.set_ylabel('Pressure (hPa)', fontsize=final_font_sizes['label'])
                yticks = [1, 10, 30]
                ax.set_yticks(yticks)
                ax.yaxis.set_major_formatter(FuncFormatter(log_sci_formatter))
                ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        
        max_p = loaded_data['True MLS OH'].pressure.max().values
        min_p = loaded_data['True MLS OH'].pressure.min().values
        axes[0].set_ylim(max_p*1.05, min_p*0.95)

    handles, labels = [], []
    if num_to_sample > 0:
        # 从第一个有内容的子图中获取图例句柄和标签
        first_ax_with_content = next((ax for ax in axes if ax.has_data()), None)
        if first_ax_with_content:
            h, l = first_ax_with_content.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in labels:
                    labels.append(label)
                    handles.append(handle)
    
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(labels), 
                   bbox_to_anchor=(0.5, 1.0), fontsize=final_font_sizes['legend'])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    data_date_obj = datetime.strptime(f'{year}-{doy}', '%Y-%j')
    formatted_date = data_date_obj.strftime('%d %b %Y')    
    fig.text(0.99, 0.98, f'Date: {formatted_date}', ha='right', va='top',
             fontsize=11, color='grey', alpha=0.7)
    
    plot_path = os.path.join(plot_dir, f'Figure2_profile_comparison_{year}_day{doy}.png')
    plt.savefig(plot_path, dpi=400)
    plt.close(fig)
    print(f"✅ Saved final OH profile comparison plot to: {plot_path}")