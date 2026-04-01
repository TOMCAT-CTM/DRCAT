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
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable # <--- 1. 导入新工具



Constant_var_path = 'Data/constant_var_data'  # CONSTANT_VAR_DATA_DIR


try:
    import cartopy.crs as ccrs
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    print("警告: 未找到 'cartopy' 。")



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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    在[0, 360]经度范围内，对稀疏数据进行加权网格化。
    """
    start_time = time.time()

    stack = data_array.stack(points=('lon', 'lat')).dropna(dim='points')
    raw_lons = stack.lon.values
    raw_lats = stack.lat.values
    raw_values = stack.values

    if len(raw_values) == 0:
        print("错误：筛选后没有剩下任何有效数据点。")
        return None, None, None

    grid_lats_centers = np.arange(-90 + lat_spacing / 2, 90, lat_spacing)
    grid_lons_centers = np.arange(0 + lon_spacing / 2, 360, lon_spacing)
    gridded_data = np.full((len(grid_lats_centers), len(grid_lons_centers)), np.nan)
    
    lat_search_radius = search_radius_factor * lat_half_width
    lon_search_radius = search_radius_factor * lon_half_width
    
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
        print(f"Info: 未能加载或处理文件 {os.path.basename(filepath)}。")
        return None

def _calculate_metrics(true_arr, pred_arr):
    """
    计算两个数组之间的RMSE、R²和SSIM指标。
    """
    if true_arr.shape != pred_arr.shape:
        return None

    valid_mask = np.isfinite(true_arr) & np.isfinite(pred_arr)
    if valid_mask.sum() < 2:  # 至少需要两个点才能计算R²
        return None
        
    v_true, v_pred = true_arr[valid_mask], pred_arr[valid_mask]
    rmse = np.sqrt(mean_squared_error(v_true, v_pred))
    r2 = r2_score(v_true, v_pred)

    true_no_nan = np.nan_to_num(true_arr)
    pred_no_nan = np.nan_to_num(pred_arr)
    
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


def plot_zonal_mean_and_anomaly_timeline(
    variable_name,
    start_day,
    end_day,
    step_days,
    plot_dir,
    OUTPUT_DIR,
    baseline_year=2021,
    target_year=2022,
    plot_args=None,
):
    """
    在同一张图中并排绘制 global zonal mean（左列）和相对异常值（右列）。
    """
    if plot_args is None:
        plot_args = {}

    fig_w = 7.25
    row_h = plot_args.get('row_h', 1.0)
    hspace = plot_args.get('hspace', 0.58)
    wspace = plot_args.get('wspace', 0.12)
    left_margin = plot_args.get('left_margin', 0.09)
    right_margin = plot_args.get('right_margin', 0.985)
    top_margin = plot_args.get('top_margin', 0.965)
    bottom_margin = plot_args.get('bottom_margin', 0.15)
    cbar_height = plot_args.get('cbar_height', 0.022)
    cbar_gap = plot_args.get('cbar_gap', 0.08)
    cmap_mean = plot_args.get('cmap_mean', 'jet')
    cmap_anomaly = plot_args.get('cmap_anomaly', 'RdBu_r')
    anomaly_abs_max = plot_args.get('anomaly_abs_max', 100)

    font_settings = {
        'title': 8,
        'label': 8,
        'ticks': 8,
        'legend_title': 8,
        'legend_text': 8,
        'offset_text': 8,
        'panel': 9,
        'cbar_label': 7,
    }

    font_path = 'Plot/fonts/MYRIADPRO-REGULAR.OTF'
    bold_font_path = 'Plot/fonts/MYRIADPRO-BOLD.OTF'
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        myriad_font = prop.get_name()
        if os.path.exists(bold_font_path):
            fm.fontManager.addfont(bold_font_path)
            myriad_bold_font = fm.FontProperties(fname=bold_font_path)
        else:
            myriad_bold_font = fm.FontProperties(weight='bold')
    else:
        print(f"Warning: Font file {font_path} not found. Using sans-serif.")
        myriad_font = 'Arial'
        myriad_bold_font = fm.FontProperties(weight='bold')

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Myriad Pro', 'Myriad', 'Arial', 'Helvetica'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.titlesize': 8,
        'figure.titlesize': 8,
        'axes.linewidth': 0.5,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
    })

    print(f"\n--- Generating Combined Zonal Mean + Anomaly Timeline for {variable_name} ---")
    print(f"Baseline year: {baseline_year}, Target year: {target_year}")

    if variable_name == 'DRCAT-Predicted OH':
        baseline_fname = f'predicted_MLS_OH_density_gridded_{baseline_year}.nc'
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

    try:
        baseline_ds = xr.open_dataset(os.path.join(OUTPUT_DIR, baseline_fname))
        target_ds = xr.open_dataset(os.path.join(OUTPUT_DIR, target_fname))
    except FileNotFoundError as e:
        print(f"Could not load data: {e}")
        return
    
    # convert units to 10^6 cm^-3
    baseline_ds = baseline_ds / 1e6
    target_ds = target_ds / 1e6

    var_name_baseline = list(baseline_ds.data_vars)[0]
    var_name_target = list(target_ds.data_vars)[0]

    plev_min = 0.9
    plev_max = 33
    days = list(range(start_day, end_day + 1, step_days))
    n_days = len(days)

    mean_data_by_day = {}
    anomaly_data_by_day = {}
    all_mean = []
    all_anomaly = []

    for day in days:
        try:
            baseline_daily = baseline_ds[var_name_baseline].sel(doy=day, method='nearest').sel(pressure=slice(plev_min, plev_max))
            target_daily = target_ds[var_name_target].sel(doy=day, method='nearest').sel(pressure=slice(plev_min, plev_max))

            baseline_zonal = baseline_daily.mean(dim='lon', skipna=True)
            target_zonal = target_daily.mean(dim='lon', skipna=True)
            anomaly = xr.where(baseline_zonal != 0, (target_zonal - baseline_zonal) / baseline_zonal * 100, np.nan)

            mean_data_by_day[day] = target_zonal
            anomaly_data_by_day[day] = anomaly
            all_mean.append(target_zonal)
            all_anomaly.append(anomaly)
        except (IndexError, KeyError):
            continue

    if not all_mean:
        print('No valid mean/anomaly data available to plot.')
        baseline_ds.close()
        target_ds.close()
        return

    mean_max = max(float(np.nanmax(d.values)) for d in all_mean)
    mean_min = min(float(np.nanmin(d.values)) for d in all_mean)
    if not np.isfinite(mean_min) or not np.isfinite(mean_max):
        mean_min, mean_max = 0.0, 1.0

    fig_h = max(2.2, row_h * n_days + 0.55)
    fig, axes = plt.subplots(
        nrows=n_days,
        ncols=2,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=True,

    )
    fig.subplots_adjust(
        left=left_margin,
        right=right_margin,
        top=top_margin,
        bottom=bottom_margin,
        hspace=hspace,
        wspace=wspace,
    )
    

    if n_days == 1:
        axes = np.array([axes])

    pcm_mean = None
    pcm_anom = None

    for i, day in enumerate(days):
        ax_mean = axes[i, 0]
        ax_anom = axes[i, 1]

        if day not in mean_data_by_day:
            ax_mean.text(0.5, 0.5, f'No data for Day {day}', ha='center', va='center', transform=ax_mean.transAxes)
            ax_anom.text(0.5, 0.5, f'No data for Day {day}', ha='center', va='center', transform=ax_anom.transAxes)
            continue

        target_zonal = mean_data_by_day[day]
        anomaly = anomaly_data_by_day[day]

        pcm_mean = target_zonal.plot.pcolormesh(
            ax=ax_mean,
            x='lat',
            y='pressure',
            cmap=cmap_mean,
            vmin=mean_min,
            vmax=mean_max,
            add_colorbar=False,
        )
        pcm_anom = anomaly.plot.pcolormesh(
            ax=ax_anom,
            x='lat',
            y='pressure',
            cmap=cmap_anomaly,
            vmin=-anomaly_abs_max,
            vmax=anomaly_abs_max,
            add_colorbar=False,
        )

        date_obj = datetime.strptime(f'{target_year} {day}', '%Y %j')
        date_str = f"{date_obj.strftime('%B')} {date_obj.day}"

        ax_mean.set_yscale('log')
        ax_anom.set_yscale('log')
        ax_mean.invert_yaxis()
        # ax_anom.invert_yaxis()
    
        ax_mean.set_title(f'Day {day} ({date_str})')
        ax_anom.set_title(f'Day {day} ({date_str})')

        ax_mean.set_ylabel('Pressure (hPa)', fontsize=font_settings['label'])
        ax_anom.set_ylabel('')
        ax_anom.tick_params(axis='y', labelleft=False)
        

        if i == n_days - 1:
            ax_mean.set_xlabel(r'Latitude (°N)', fontsize=font_settings['label'])
            ax_anom.set_xlabel(r'Latitude (°N)', fontsize=font_settings['label'])
        else:
            ax_mean.set_xlabel('')
            ax_anom.set_xlabel('')

        ax_mean.tick_params(axis='both', labelsize=font_settings['ticks'])
        ax_anom.tick_params(axis='both', labelsize=font_settings['ticks'])

        if i == 0:
            ax_mean.text(0.02, 1.16, 'OH Mean', transform=ax_mean.transAxes, fontsize=font_settings['panel'], fontproperties=myriad_bold_font, va='top', ha='left')
            ax_anom.text(0.02, 1.16, 'OH Anomaly', transform=ax_anom.transAxes, fontsize=font_settings['panel'], fontproperties=myriad_bold_font, va='top', ha='left')

    if pcm_mean is not None:
        left_pos = axes[-1, 0].get_position()
        cax_mean = fig.add_axes([
            left_pos.x0,
            bottom_margin - cbar_gap,
            left_pos.width,
            cbar_height,
        ])
        cbar_mean = fig.colorbar(
            pcm_mean,
            cax=cax_mean,
            orientation='horizontal',
        )
        cbar_mean.set_label(r'Density (10$^{6}$ cm$^{-3}$)', fontsize=font_settings['cbar_label'])
        cbar_mean.ax.tick_params(labelsize=font_settings['ticks'])

    if pcm_anom is not None:
        right_pos = axes[-1, 1].get_position()
        cax_anom = fig.add_axes([
            right_pos.x0,
            bottom_margin - cbar_gap,
            right_pos.width,
            cbar_height,
        ])
        cbar_anom = fig.colorbar(
            pcm_anom,
            cax=cax_anom,
            orientation='horizontal',
        )
        cbar_anom.set_label('Relative Anomaly (%)', fontsize=font_settings['cbar_label'])
        cbar_anom.ax.tick_params(labelsize=font_settings['ticks'])

    # plt.tight_layout()

    out_name = f"{variable_name.replace(' ', '_')}_mean_anomaly_timeline_{target_year}-{baseline_year}_day{start_day}-{end_day}.pdf"
    out_path = os.path.join(plot_dir, out_name)
    plt.savefig(out_path, dpi=600)
    plt.close(fig)

    baseline_ds.close()
    target_ds.close()
    print(f"Saved combined mean+anomaly timeline plot for {variable_name}: {out_path}")


def plot_zonal_mean_comparison_sci(year, day_of_year, plot_dir, output_dir):
    """
    生成纬向平均OH浓度对比图
    """
    print(f"\n--- 正在为 {year} 年, 第 {day_of_year} 天生成对比图---")

    # --- 1. 数据加载 --- 
    data_sources = {
        'ML-Predicted OH': f'predicted_MLS_OH_density_gridded_{year}.nc',
        'Calculated SSA-OH': f'calculated_SSA_OH_density_gridded_{year}.nc',
        # 'Predicted SSA-OH': f'predicted_SSA_OH_density_gridded_{year}.nc',
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

    # set unit = '10^6 cm^-3' for all datasets
    for key in loaded_data:
        loaded_data[key] = loaded_data[key] / 1e6

    # --- 2. 指标计算 ---
    comparisons = {
        "Pred. MLS vs True": ('ML-Predicted OH', 'True MLS OH'),
        "Calc. SSA vs True": ('Calculated SSA-OH', 'True MLS OH'),
        "Pred. SSA vs True": ('Predicted SSA-OH', 'True MLS OH'),
        "TOMCAT vs True": ('TOMCAT OH', 'True MLS OH')
    }
    
    comp_items = list(comparisons.items())
    mid_point = (len(comp_items) + 1) // 2 
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

    max_len_col1 = max(len(s) for s in col1_lines) if col1_lines else 0
    formatted_lines = []
    for i in range(len(col1_lines)):
        line1 = col1_lines[i]
        line2 = col2_lines[i] if i < len(col2_lines) else ""
        formatted_lines.append(f"{line1:<{max_len_col1}}    {line2}")
    metrics_text_aligned = "\n".join(formatted_lines)
    

    # 3. 绘图初始化
    # plt.style.use('seaborn-v0_8-paper')
    # --- Science Advances标准 ---
    font_settings = {
        'title': 8,
        'label': 8,
        'ticks': 8,
        'legend_title':8,
        'legend_text': 8,
        'offset_text': 8,
        'panel': 9,
        'cbar_label': 7,
    }
    # load regular font and bold font
    font_path = 'Plot/fonts/MYRIADPRO-REGULAR.OTF'
    bold_font_path = 'Plot/fonts/MYRIADPRO-BOLD.OTF'
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        # 获取加载后的字体真实名称（通常是 'Myriad Pro'）
        prop = fm.FontProperties(fname=font_path)
        myriad_font = prop.get_name()
        if os.path.exists(bold_font_path):
            fm.fontManager.addfont(bold_font_path)
            myriad_bold_font = fm.FontProperties(fname=bold_font_path)
    else:
        print(f"Warning: Font file {font_path} not found. Using sans-serif.")
        myriad_font = 'Arial'
        myriad_bold_font = fm.FontProperties(weight='bold')   

    plt.rcParams.update({
        'font.family': 'sans-serif', 
        'font.sans-serif': ['Myriad Pro', 'Myriad', 'Arial', 'Helvetica'], # 首选 Myriad
        'pdf.fonttype': 42,      
        'ps.fonttype': 42,
        'axes.labelsize': 8,     # 轴标签字号设为 8pt (期刊默认字号)
        'xtick.labelsize': 8,    # 刻度字号设为 6pt (期刊允许的最小字号)
        'ytick.labelsize': 8,
        'axes.titlesize': 8,     # 子图标题设为 8pt
        'figure.titlesize': 8,
        'axes.linewidth': 0.5,   # 线宽 0.5pt (需大于期刊规定的 0.28pt)
        'xtick.major.size': 2.5,   # X轴主刻度线长度缩短 (默认约 3.5-4)
        'ytick.major.size': 2.5,   # Y轴主刻度线长度缩短
        'xtick.major.width': 0.5,  # 刻度线粗细与坐标轴线宽保持一致
        'ytick.major.width': 0.5,  # 刻度线粗细与坐标轴线宽保持一致

    })

    # --- 4. 绘图 ---
    plot_order = ['True MLS OH', 'ML-Predicted OH', 'Calculated SSA-OH', 'TOMCAT OH']
    plot_data = {k: loaded_data[k] for k in plot_order if k in loaded_data}
    num_plots = len(plot_data)
    if num_plots == 0: return

    fig, axes = plt.subplots(
        nrows=1, ncols=num_plots, 
        figsize=(7.25, 3),         # 修改为符合期刊的双栏宽 7.25 in
        sharex=True, sharey=True,
        gridspec_kw={'bottom': 0.2} 
    )
    if num_plots == 1: axes = [axes]
    
    vmax = loaded_data['True MLS OH'].max()
    cbar_label = r'OH density (10$^{6}$ cm$^{-3}$)'

    for i, (title, data) in enumerate(plot_data.items()):
        ax = axes[i]
        pcm = data.plot.pcolormesh(
            ax=ax, x='lat', y='pressure', cmap='turbo',
            vmin=0, vmax=25, add_colorbar=False
        )
        if title == 'True MLS OH':
            ax.set_title('MLS OH', pad=3)
        else:
            ax.set_title(title, pad=3)
        ax.set_xlabel(''); ax.set_ylabel('')
        labels=['A', 'B', 'C', 'D']
        ax.text(0.02, 1.06, labels[i], transform=ax.transAxes, fontsize=font_settings['panel'], fontproperties=myriad_bold_font, va='top', ha='left')


        # 添加指标到对比子图下方
        if title != 'True MLS OH':
            metrics = _calculate_metrics(loaded_data['True MLS OH'].values, data.values)
            if metrics:
                metric_str = f"RMSE: {metrics['rmse']:.2f}  R²: {metrics['r2']:.3f}  SSIM: {metrics['ssim']:.3f}"
                ax.text(0.5, -0.2, metric_str, ha='center', va='top', transform=ax.transAxes, fontsize=7)
                
    # 修改全局坐标轴标签字体大小为 8pt
    fig.supxlabel('Latitude (°N)', fontsize=8, y=0.08)
    fig.supylabel('Pressure (hPa)', fontsize=8, x=0.02,y=0.6)
    
    axes[0].set_yscale('log')
    true_data = loaded_data['True MLS OH']
    max_p = true_data.pressure.max().values
    min_p = true_data.pressure.min().values
    axes[0].set_ylim(max_p, min_p)  

    fig.subplots_adjust(left=0.08, right=0.9, top=0.95, bottom=0.25, wspace=0.095)
    # fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    cbar_ax = fig.add_axes([0.915, 0.25, 0.012, 0.7])
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    cbar.set_label(cbar_label, size=8)
    cbar.ax.tick_params(labelsize=6) # colorbar 刻度设为 6pt
    


    # --- 添加日期文本 ---
    # data_date_obj = datetime.strptime(f'{year}-{day_of_year}', '%Y-%j')
    # formatted_date = data_date_obj.strftime('%d %b %Y')    
    # fig.text(
    #     0.9, 0.95,
    #     f'Date: {formatted_date}', 
    #     ha='right', va='bottom',
    #     fontsize=6, color='grey', alpha=0.7 # 字体修改为 6pt
    # )

    # 保存图片
    plot_path = os.path.join(plot_dir, f'Figure2_zonal_mean_comparison_{year}_day{day_of_year}_sci_compact.pdf')
    plt.savefig(plot_path, dpi=600)
    plt.close(fig)
    print(f"已保存对比图: {plot_path}")


def plot_zonal_mean_difference_sci(year, day_of_year, plot_dir, output_dir):
    """
    生成纬向平均OH浓度差异图。
    绘制4列：
    - 第1列：MLS Obs OH（绝对值）
    - 第2列：DRCAT predicted OH - MLS Obs OH
    - 第3列：SSA OH - MLS Obs OH
    - 第4列：TOMCAT OH - MLS Obs OH
    """
    print(f"\n--- 正在为 {year} 年, 第 {day_of_year} 天生成差异对比图 ---")
    

    # --- 1. 数据加载 ---
    data_sources = {
        'ML-Predicted OH': f'predicted_MLS_OH_density_gridded_{year}.nc',
        'Calculated SSA-OH': f'calculated_SSA_OH_density_gridded_{year}.nc',
        # 'Predicted SSA-OH': f'predicted_SSA_OH_density_gridded_{year}.nc',
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

    # 转换单位为 10^6 cm^-3
    for key in loaded_data:
        loaded_data[key] = loaded_data[key] / 1e6

    # --- 2. 计算差异数据 ---
    true_mls = loaded_data['True MLS OH']
    differences = {}
    plot_titles = ['MLS OH']
    plot_data_list = [true_mls]
    
    # DRCAT 差异
    if 'ML-Predicted OH' in loaded_data:
        diff = loaded_data['ML-Predicted OH'] - true_mls
        differences['DRCAT predicted'] = diff
        plot_data_list.append(diff)
        plot_titles.append('DRCAT − MLS OH')
    
    # SSA 差异
    if 'Calculated SSA-OH' in loaded_data:
        diff = loaded_data['Calculated SSA-OH'] - true_mls
        differences['SSA'] = diff
        plot_data_list.append(diff)
        plot_titles.append('SSA − MLS OH')
    
    # TOMCAT 差异
    if 'TOMCAT OH' in loaded_data:
        diff = loaded_data['TOMCAT OH'] - true_mls
        differences['TOMCAT'] = diff
        plot_data_list.append(diff)
        plot_titles.append('TOMCAT − MLS OH')

    if len(plot_data_list) < 2:
        print("错误: 无足够数据绘制差异图。已跳过绘图。")
        return


    # 3. 绘图初始化
    # --- Science Advances标准 ---
    font_settings = {
        'title': 8,
        'label': 8,
        'ticks': 8,
        'legend_title':8,
        'legend_text': 8,
        'offset_text': 8,
        'panel': 9,
        'cbar_label': 7,
    }
    # load regular font and bold font
    font_path = 'Plot/fonts/MYRIADPRO-REGULAR.OTF'
    bold_font_path = 'Plot/fonts/MYRIADPRO-BOLD.OTF'
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        # 获取加载后的字体真实名称（通常是 'Myriad Pro'）
        prop = fm.FontProperties(fname=font_path)
        myriad_font = prop.get_name()
        if os.path.exists(bold_font_path):
            fm.fontManager.addfont(bold_font_path)
            myriad_bold_font = fm.FontProperties(fname=bold_font_path)
    else:
        print(f"Warning: Font file {font_path} not found. Using sans-serif.")
        myriad_font = 'Arial'
        myriad_bold_font = fm.FontProperties(weight='bold')   

    plt.rcParams.update({
        'font.family': 'sans-serif', 
        'font.sans-serif': ['Myriad Pro', 'Myriad', 'Arial', 'Helvetica'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.titlesize': 8,
        'figure.titlesize': 8,
        'axes.linewidth': 0.5,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
    })

    # --- 4. 绘图 ---
    num_plots = len(plot_data_list)
    
    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_plots,
        figsize=(7.25, 2.7),
        sharex=True,
        sharey=True
    )
    if num_plots == 1:
        axes = [axes]
    
    pcm_abs = None  # 存储第一个子图的 pcm
    pcm_diff = None  # 存储差异子图的 pcm
    
    true_data = loaded_data['True MLS OH']
    max_p = true_data.pressure.max().values
    min_p = true_data.pressure.min().values
    
    for i, (ax, title, data) in enumerate(zip(axes, plot_titles, plot_data_list)):
        labels = ['A', 'B', 'C', 'D']
        ax.text(0.02, 1.06, labels[i], transform=ax.transAxes, fontsize=font_settings['panel'], fontproperties=myriad_bold_font, va='top', ha='left')

        if i == 0:
            pcm_abs = data.plot.pcolormesh(
                ax=ax, x='lat', y='pressure', cmap='turbo',
                vmin=0, vmax=25, add_colorbar=False
            )
            ax.set_title(title, pad=3)
            ax.set_ylabel('Pressure (hPa)', fontsize=8)
            yticks = [1, 10, 30, 100]
            ax.set_yticks(yticks)
            ax.yaxis.set_major_formatter(FuncFormatter(log_sci_formatter))
        else:

            diff_max = max(abs(data.min()), abs(data.max()))
            if np.isnan(diff_max) or diff_max == 0:
                diff_max = 5
            
            pcm_diff = data.plot.pcolormesh(
                ax=ax, x='lat', y='pressure', cmap='RdBu_r',
                vmin=-diff_max, vmax=diff_max, add_colorbar=False
            )
            ax.set_title(title, pad=3)
            
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)
        
        ax.set_xlabel('')
        
        ax.set_yscale('log')
        ax.set_ylim(max_p, min_p)

    fig.supxlabel('Latitude (°N)', fontsize=8, y=0.03)

    fig.subplots_adjust(left=0.08, right=0.85, top=0.92, bottom=0.15, wspace=0.08)

    if pcm_abs is not None:
        cbar_ax_abs = fig.add_axes([0.865, 0.15, 0.01, 0.77])
        cbar_abs = fig.colorbar(pcm_abs, cax=cbar_ax_abs)
        cbar_abs.set_label(r'$\mathregular{OH\;density\;}$' + r'$(10^6\;cm^{-3})$', 
                          size=7, labelpad=3)
        cbar_abs.ax.tick_params(labelsize=7)
    
    if pcm_diff is not None:
        cbar_ax_diff = fig.add_axes([0.93, 0.15, 0.01, 0.77])
        cbar_diff = fig.colorbar(pcm_diff, cax=cbar_ax_diff)
        cbar_diff.set_label(r'$\mathregular{OH\;difference\;}$' + r'$(10^6\;cm^{-3})$', 
                           size=7, labelpad=3)
        cbar_diff.ax.tick_params(labelsize=7)

    # 保存图片
    plot_path = os.path.join(plot_dir, f'Figure2_Supp_zonal_mean_difference_{year}_day{day_of_year}_sci_compact.pdf')
    plt.savefig(plot_path, dpi=600)
    plt.close(fig)
    print(f"已保存差异对比图: {plot_path}")


def plot_comprehensive_error_analysis(years, OUTPUT_DIR, plot_dir):
    """
    生成一个误差分析图表，比较ML预测的OH、SSA计算的OH和TOMCAT OH与真实MLS OH之间的误差指标（RMSE、R²、SSIM）。
    """
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
            # change unit to 10^6 cm^-3
            ds_true[true_var] = ds_true[true_var] / 1e6
            ds_pred[pred_var] = ds_pred[pred_var] / 1e6
            ds_calc[calc_var] = ds_calc[calc_var] / 1e6
            ds_tomcat[tomcat_var] = ds_tomcat[tomcat_var] / 1e6 

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
      
                for p_level in pressure_levels:
                    true_at_p = true_zonal.sel(pressure=p_level,method='nearest').values
                    pred_at_p = pred_zonal.sel(pressure=p_level,method='nearest').values
                    calc_at_p = calc_zonal.sel(pressure=p_level,method='nearest').values
                    tomcat_at_p = tomcat_zonal.sel(pressure=p_level,method='nearest').values # <-- 新增

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
    
    df_pressure['pressure_str'] = df_pressure['pressure'].apply(lambda x: f"{float(x):.2f}")

    #### 绘图设置
    plt.style.use('seaborn-v0_8-paper')

    # load regular font and bold font
    font_path = 'Plot/fonts/MYRIADPRO-REGULAR.OTF'
    bold_font_path = 'Plot/fonts/MYRIADPRO-BOLD.OTF'
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        # 获取加载后的字体真实名称（通常是 'Myriad Pro'）
        prop = fm.FontProperties(fname=font_path)
        myriad_font = prop.get_name()
        if os.path.exists(bold_font_path):
            fm.fontManager.addfont(bold_font_path)
            myriad_bold_font = fm.FontProperties(fname=bold_font_path)
    else:
        print(f"Warning: Font file {font_path} not found. Using sans-serif.")
        myriad_font = 'Arial'
        myriad_bold_font = fm.FontProperties(weight='bold')    


    font_settings = {
        'title': 8,        
        'label': 8,        
        'ticks': 8,        
        'legend_title': 8, 
        'legend_text': 8,  
        'offset_text': 8,
        'panel':9,
        'line_width':0.8,
        'box_line_width':0.5
    }

    # 全局字体更新：首选 Myriad，强制使用 sans-serif，关闭次要刻度
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [myriad_font, 'Myriad', 'Arial', 'Helvetica'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': font_settings['ticks'],
        'axes.linewidth': 0.3,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        # 'xtick.minor.size': 0,
        # 'ytick.minor.size': 0,
    })
    # --- 4. 绘图 ---
    
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.5), gridspec_kw={'height_ratios': [2, 2]})

    # --- Palette including TOMCAT ---
    palette = {
        'ML-Predicted OH': 'royalblue',
        'Calculated SSA-OH': 'darkorange',
        'TOMCAT OH': 'forestgreen'
    }
    

    # --- Plot 1: RMSE Distribution ---
    sns.histplot(data=df_daily, x='rmse', hue='type', bins=200, kde=True, ax=axes[0, 0], palette=palette, stat='probability',line_kws={'linewidth':font_settings['line_width']})
    axes[0, 0].set_xlim(0.9, 4)
    axes[0, 0].set_title('Daily RMSE Distribution', fontsize=font_settings['title'])
    axes[0, 0].text(0.02, 0.95, 'A', transform=axes[0, 0].transAxes, fontsize=font_settings['panel'], fontproperties=myriad_bold_font, va='top', ha='left')
    axes[0, 0].set_xlabel(r'RMSE (10$^{6}$ molec. cm$^{-3}$)', fontsize=font_settings['label'])
    axes[0, 0].set_ylabel('Normalized Counts', fontsize=font_settings['label'])
    axes[0, 0].tick_params(axis='both', labelsize=font_settings['ticks']) # Control tick label size
    # axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    # --- Plot 2: R² Distribution ---
    sns.histplot(data=df_daily, x='r2', hue='type', bins=200, kde=True, ax=axes[0, 1], palette=palette,stat='probability',line_kws={'linewidth':font_settings['line_width']})
    axes[0, 1].set_xlim(0.8, 1)
    axes[0, 1].set_title(r'Daily $R^2$ Distribution', fontsize=font_settings['title'])
    axes[0, 1].text(0.02, 0.95, 'B', transform=axes[0, 1].transAxes, fontsize=font_settings['panel'], fontproperties=myriad_bold_font, va='top', ha='left')

    axes[0, 1].set_xlabel(r'Coefficient of Determination ($R^2$)', fontsize=font_settings['label'])
    axes[0, 1].set_ylabel('Normalized Counts', fontsize=font_settings['label'])
    axes[0, 1].tick_params(axis='both', labelsize=font_settings['ticks']) # Control tick label size
    # axes[0, 1].grid(True, linestyle='--', alpha=0.6)

    boxplot_kwargs = {
        "boxprops": {'alpha': 0.7, 'linewidth':font_settings['box_line_width']},
        "whiskerprops": {'linewidth': font_settings['box_line_width']},
        "capprops": {'linewidth': font_settings['box_line_width']},
        "medianprops": {'linewidth': font_settings['box_line_width']}
    }

    # --- Plot 3: Monthly RMSE Boxplot ---
    sns.boxplot(data=df_daily, x='month', y='rmse', hue='type', ax=axes[1, 0], palette=palette, showfliers=False, **boxplot_kwargs)
    # axes[1, 0].set_ylim(0.9, 2.75)
    # axes[1, 0].set_xlim(0.5, 12.5)
    axes[1, 0].set_title('Monthly RMSE Statistics', fontsize=font_settings['title'])
    axes[1, 0].text(0.02, 0.95, 'C', transform=axes[1, 0].transAxes, fontsize=font_settings['panel'], fontproperties=myriad_bold_font, va='top', ha='left')

    axes[1, 0].set_xlabel('Month', fontsize=font_settings['label'])
    axes[1, 0].set_ylabel(r'Daily RMSE (10$^{6}$ molec. cm$^{-3})$', fontsize=font_settings['label'])
    # 设置月份刻度
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    axes[1, 0].set_xticks(range(0, 12))
    axes[1, 0].set_xticklabels(month_labels)
    axes[1, 0].tick_params(axis='both', labelsize=font_settings['ticks']) # Control tick label size
    # axes[1, 0].grid(True, axis='y', linestyle='--', alpha=0.6)
    import matplotlib.ticker as mticker
    # --- Plot 4: Pressure Level RMSE Boxplot (Horizontal) ---
    sns.boxplot(data=df_pressure, y='pressure_str', x='rmse', hue='type', ax=axes[1, 1], palette=palette, orient='h', showfliers=False, **boxplot_kwargs)
    axes[1, 1].set_title('RMSE by Pressure Level', fontsize=font_settings['title'])
    axes[1, 1].text(0.02, 0.95, 'D', transform=axes[1, 1].transAxes, fontsize=font_settings['panel'], fontproperties=myriad_bold_font, va='top', ha='left')

    axes[1, 1].set_ylabel('Pressure (hPa)', fontsize=font_settings['label'])
    axes[1, 1].set_xlabel(r'RMSE (10$^{6}$ molec. cm$^{-3}$)', fontsize=font_settings['label'])
    axes[1, 1].tick_params(axis='both', labelsize=font_settings['ticks']) # Control tick label size

    
    axes[0, 0].xaxis.get_offset_text().set_size(font_settings['offset_text']) # <-- New: Adjust size of '1e6'
    axes[0, 1].xaxis.get_offset_text().set_size(font_settings['offset_text']) # <-- New: Adjust size of '1e6'
    axes[1, 0].yaxis.get_offset_text().set_size(font_settings['offset_text']) # <-- New: Adjust size of '1e6'
    axes[1, 1].xaxis.get_offset_text().set_size(font_settings['offset_text']) # <-- New: Adjust size of '1e6'   

    # 在删除legend前先保存handles和labels
    handles, labels = axes[1, 1].get_legend_handles_labels()
    # change display labels to match the plot titles
    label_mapping = {
        'ML-Predicted OH': 'DRCAT OH',
        'Calculated SSA-OH': 'SSA OH',
        'TOMCAT OH': 'TOMCAT OH'
    }
    labels = [label_mapping.get(label, label) for label in labels]
    # remove legend in subplots
    for ax in axes.flat:
        if ax.get_legend():
            ax.get_legend().remove()
    
    if handles and labels:
        fig.legend(handles=handles, labels=labels, loc='lower center', ncol=3,
                   bbox_to_anchor=(0.5, 0.48), fontsize=font_settings['legend_text'],
                   frameon=False)
    
    plt.subplots_adjust(wspace=0.25, hspace=0.4, left=0.08, right=0.97, top=0.97, bottom=0.08)

    plot_path = os.path.join(plot_dir, f'Figure3_comprehensive_error_analysis_{years[0]}-{years[-1]}.pdf')
    plt.savefig(plot_path, dpi=600)
    plt.close(fig)
    print(f"已保存综合误差分析图: {plot_path}")


def plot_hunga_tonga_analysis_log(source, plot_year, baseline_year, doy, plot_dir, OUTPUT_DIR, pressure_level=10.0):
    """
    绘制Hunga-Tonga火山爆发影响分析图 
    """
    print(f"\n--- Generating Hunga-Tonga Eruption Analysis for doy {doy} ---")
    
    # 局部导入以确保LogNorm可用
    from matplotlib.colors import LogNorm
    import matplotlib.ticker as mticker

    def _load_and_process_day(variable_name, year, day_of_year, output_dir):
        if variable_name == 'DRCAT-Predicted OH':
            fname = f'predicted_MLS_OH_density_gridded_{year}.nc'
        elif variable_name.upper() == 'H2O':
            fname = f'H2O_gridded_{year}.nc'
        elif variable_name == 'SSA OH':
            fname = f'calculated_SSA_OH_density_gridded_{year}.nc'
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

    title_fontsize = 6.5
    
    # 1. 加载和计算数据
    h2o_plot_data = _load_and_process_day('H2O', plot_year, doy, Constant_var_path)*1e6 # 转换为 ppmv
    h2o_base_data = _load_and_process_day('H2O', baseline_year, doy, Constant_var_path)*1e6 # 转换为 ppmv
    oh_plot_data = _load_and_process_day(source, plot_year, doy, OUTPUT_DIR)*1e-6 # 转换为 10^6 cm^-3
    oh_base_data = _load_and_process_day(source, baseline_year, doy, OUTPUT_DIR)*1e-6 # 转换为 10^6 cm^-3
    
    if any(data is None for data in [h2o_plot_data, h2o_base_data, oh_plot_data, oh_base_data]):
        print("Error: Missing necessary data. Aborting plot."); return
        
    h2o_zonal_plot = h2o_plot_data.mean(dim='lon', skipna=True)
    h2o_zonal_anomaly = (h2o_zonal_plot - h2o_base_data.mean(dim='lon', skipna=True)) / h2o_base_data.mean(dim='lon', skipna=True) * 100
    oh_zonal_plot = oh_plot_data.mean(dim='lon', skipna=True)
    oh_zonal_anomaly = (oh_zonal_plot - oh_base_data.mean(dim='lon', skipna=True)) / oh_base_data.mean(dim='lon', skipna=True) * 100
    
    # --- 准备插值图数据 ---
    oh_map_plot_raw = oh_plot_data.sel(pressure=pressure_level, method='nearest')
    h2o_map_plot_raw = h2o_plot_data.sel(pressure=pressure_level, method='nearest')

    # 调用加权网格化函数
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

    # 3. 绘图初始化
    plt.style.use('seaborn-v0_8-paper')
    # --- Science Advances标准 ---
    font_settings = {
        'title': 8,
        'label': 8,
        'ticks': 7,
        'legend_title':8,
        'legend_text': 8,
        'offset_text': 8,
        'panel': 9,
        'cbar_label': 7,
    }
    # load regular font and bold font
    font_path = 'Plot/fonts/MYRIADPRO-REGULAR.OTF'
    bold_font_path = 'Plot/fonts/MYRIADPRO-BOLD.OTF'
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        # 获取加载后的字体真实名称（通常是 'Myriad Pro'）
        prop = fm.FontProperties(fname=font_path)
        myriad_font = prop.get_name()
        if os.path.exists(bold_font_path):
            fm.fontManager.addfont(bold_font_path)
            myriad_bold_font = fm.FontProperties(fname=bold_font_path)
    else:
        print(f"Warning: Font file {font_path} not found. Using sans-serif.")
        myriad_font = 'Arial'
        myriad_bold_font = fm.FontProperties(weight='bold')    


    # 全局字体更新：首选 Myriad，强制使用 sans-serif，关闭次要刻度
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Myriad Pro', 'Myriad', 'Arial', 'Helvetica'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': font_settings['ticks'],
        'axes.linewidth': 0.5,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.size': 0.5,
        'ytick.minor.size': 0.5,
    })

    fig = plt.figure(figsize=(7.25, 4.1))
    fig.subplots_adjust(left=0.06, right=1, bottom=0.08, top=0.95)

    # 布局设置
    gs_outer = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.18)
    gs_left = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_outer[0],
                                               width_ratios=[1, 1], wspace=0.25, hspace=0.28)
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[1],
                                                hspace=0.28)

    axes = np.empty((2, 3), dtype=object)
    axes[0, 0] = fig.add_subplot(gs_left[0, 0])
    axes[1, 0] = fig.add_subplot(gs_left[1, 0])
    axes[0, 1] = fig.add_subplot(gs_left[0, 1])
    axes[1, 1] = fig.add_subplot(gs_left[1, 1])

    projection = ccrs.PlateCarree(central_longitude=180) if CARTOPY_AVAILABLE else None
    axes[0, 2] = fig.add_subplot(gs_right[0, 0], projection=projection)
    axes[1, 2] = fig.add_subplot(gs_right[1, 0], projection=projection)
    
    # --- Col 1 & 2: Zonal Mean Plots ---
    zonal_plot_bounds = [0.0, 0.03, 0.98, 0.98]
    for i in range(2):
        for j in range(2):
            parent_ax = axes[i, j]
            parent_ax.axis('off')
            ax = parent_ax.inset_axes(zonal_plot_bounds)
            divider = make_axes_locatable(ax)
            cax_for_colorbar = divider.append_axes("right", size="4.5%", pad=0.05)
            
            if j == 0 and i == 0:
                data = h2o_zonal_plot
                mappable = ax.pcolormesh(data.lat, data.pressure, data, cmap='turbo', vmin=0, vmax=18)
                cb = fig.colorbar(mappable, cax=cax_for_colorbar)
                cb.ax.set_title(r'H2O(ppmv)', fontsize=font_settings['legend_title'], pad=3)
                ax.set_title('H2O Zonal Mean', fontsize=title_fontsize, pad=2)
            elif j == 0 and i == 1:
                data = h2o_zonal_anomaly
                mappable = ax.pcolormesh(data.lat, data.pressure, data, cmap='RdBu_r', vmin=-100, vmax=100)
                cb = fig.colorbar(mappable, cax=cax_for_colorbar)
                cb.ax.set_title(r'Anomaly(%)', fontsize=font_settings['legend_title'], pad=3)
                ax.set_title('H2O Relative Anomaly', fontsize=title_fontsize, pad=2)
            elif j == 1 and i == 0:
                data = oh_zonal_plot
                mappable = ax.pcolormesh(data.lat, data.pressure, data, cmap='turbo', vmin=0, vmax=25)
                cb = fig.colorbar(mappable, cax=cax_for_colorbar)
                cb.ax.set_title(r'Density(10$^{6}$ cm$^{-3}$)', fontsize=font_settings['legend_title'], pad=3)
                ax.set_title('OH Zonal Mean', fontsize=title_fontsize, pad=2)
            elif j == 1 and i == 1:
                data = oh_zonal_anomaly
                mappable = ax.pcolormesh(data.lat, data.pressure, data, cmap='RdBu_r', vmin=-100, vmax=100)
                cb = fig.colorbar(mappable, cax=cax_for_colorbar)
                cb.ax.set_title(r'Anomaly(%)', fontsize=font_settings['legend_title'], pad=4)
                
                ax.set_title('OH Relative Anomaly', fontsize=title_fontsize, pad=2)

            cb.ax.tick_params(labelsize=font_settings['cbar_label'],pad=1.5)
            # set cb.ax.title font size
            cb.ax.title.set_fontsize(6)

            ax.set_yscale('log')
            max_p = data.pressure.max().values
            min_p = data.pressure.min().values
            ax.set_ylim(max_p, min_p)
            ax.tick_params(axis='both', which='major', labelsize=font_settings['ticks'])
            ax.set_xticks([-90, -45, 0, 45, 90])
            ax.set_ylabel('Pressure (hPa)' if j == 0 else '', fontsize=font_settings['label'])
            ax.set_xlabel('Latitude (°N)' if i == 1 else '', fontsize=font_settings['label'])
            if j == 1: plt.setp(ax.get_yticklabels(), visible=False)

            # add panel label
            labels=['A','B','D','E']
            ax.text(0.02, 1.07, labels[i*2 + j], transform=ax.transAxes, fontsize=font_settings['panel'], fontproperties=myriad_bold_font, va='top', ha='left')

            # 在 C、D 子图中添加红色圆圈，标记增强区域
            if j == 1:
                highlight_circle = plt.Circle(
                    (0.42, 0.12),
                    0.15,
                    transform=ax.transAxes,
                    fill=False,
                    edgecolor='red',
                    linewidth=1.2,
                    zorder=8,
                    alpha=0.5
                )
                ax.add_patch(highlight_circle)


    # --- Col 3: Interpolated Global Maps ---
    ax_oh_map = axes[0, 2]
    ax_h2o_map = axes[1, 2]

    if CARTOPY_AVAILABLE:
        ax_oh_map.set_aspect('auto')
        ax_h2o_map.set_aspect('auto')

    # 1. 设置固定范围 (避开 0，设为 1.0 到 90.0)
    OH_MAP_VMIN = 1
    OH_MAP_VMAX = 80.0 
    oh_map_gridded[oh_map_gridded<OH_MAP_VMIN] = OH_MAP_VMIN  

    custom_oh_levels = np.logspace(np.log10(OH_MAP_VMIN), np.log10(OH_MAP_VMAX), 30)

    # 3. 绘制 Contourf

    cf_oh = ax_oh_map.contourf(
        lon_centers_oh, lat_centers_oh, oh_map_gridded, 
        cmap='YlGnBu', 
        alpha=0.9,
        norm=LogNorm(vmin=OH_MAP_VMIN, vmax=OH_MAP_VMAX), # 应用对数归一化
        levels=custom_oh_levels,
        extend='both', # 小于1显示深蓝，大于90显示深红
        transform=ccrs.PlateCarree() if CARTOPY_AVAILABLE else None
    )

    # 4. 设置 Colorbar
    cb_oh = fig.colorbar(cf_oh, ax=ax_oh_map, pad=0.02)
    # Put the colorbar label on top instead of the side.
    cb_oh.ax.set_title(r'Density (10$^{6}$ cm$^{-3}$)',
                       fontsize=6, pad=4)
    
    target_ticks = [1, 5, 10, 20, 40,60, 80]
    cb_oh.set_ticks(target_ticks)
    cb_oh.set_ticklabels([str(t) for t in target_ticks])
    cb_oh.ax.tick_params(labelsize=font_settings['cbar_label'])

    ax_oh_map.set_title(f'Interpolated OH Global Map ({pressure_level} hPa)', fontsize=title_fontsize, pad=2)
    ax_oh_map.text(0.02, 1.07, 'C', transform=ax_oh_map.transAxes, fontsize=font_settings['panel'], fontproperties=myriad_bold_font, va='top', ha='left')

    # ==========================================================================
    # H2O Map 
    # ==========================================================================
    oh_map_gridded=h2o_map_gridded
    # 1. 设置固定范围 (避开 0，设为 1.0 到 90.0)
    OH_MAP_VMIN = 5
    OH_MAP_VMAX = 50.0 
    oh_map_gridded[oh_map_gridded<OH_MAP_VMIN] = OH_MAP_VMIN  # 将低于最小值的部分设为最小值，避免LogNorm报错   
    custom_oh_levels = np.logspace(np.log10(OH_MAP_VMIN), np.log10(OH_MAP_VMAX), 30)

    # 3. 绘制 Contourf
    cf_h2o = ax_h2o_map.contourf(
        lon_centers_h2o, lat_centers_h2o, oh_map_gridded, 
        cmap='YlGnBu', 
        alpha=0.9,
        norm=LogNorm(vmin=OH_MAP_VMIN, vmax=OH_MAP_VMAX), # 应用对数归一化
        levels=custom_oh_levels,
        extend='both', # 小于1显示深蓝，大于90显示深红
        transform=ccrs.PlateCarree() if CARTOPY_AVAILABLE else None
    )

    # 4. 设置 Colorbar
    cb_h2o = fig.colorbar(cf_h2o, ax=ax_h2o_map, pad=0.02)
    # Put the colorbar label on top instead of the side.
    cb_h2o.ax.set_title(r'H2O (ppmv)', fontsize=6, pad=4)


    target_ticks =  [5, 10, 20, 30,40, 50]
    cb_h2o.set_ticks(target_ticks)
    cb_h2o.set_ticklabels([str(t) for t in target_ticks])
    cb_h2o.ax.tick_params(labelsize=font_settings['cbar_label'])

    ax_h2o_map.set_title(f'Interpolated H2O Global Map ({pressure_level} hPa)', fontsize=title_fontsize, pad=2)
    ax_h2o_map.text(0.02, 1.07, 'F', transform=ax_h2o_map.transAxes, fontsize=font_settings['panel'], fontproperties=myriad_bold_font, va='top', ha='left')

    # 在 E、F 子图中标注火山爆发位置与羽流方向
    eruption_lat = -20.536
    eruption_lon = -175.382
    map_transform = ccrs.PlateCarree() if CARTOPY_AVAILABLE else ax_oh_map.transData
    if not CARTOPY_AVAILABLE and eruption_lon < 0:
        eruption_lon = eruption_lon + 360.0

    for idx, map_ax in enumerate([ax_oh_map, ax_h2o_map]):
        # 火山爆发位置：红色星标 + 文本
        map_ax.plot(
            eruption_lon,
            eruption_lat,
            marker='*',
            markersize=8,
            color='red',
            markeredgecolor='darkred',
            markeredgewidth=0.4,
            transform=map_transform,
            zorder=9,
            alpha=0.8,
        )
        if idx==1:  
            map_ax.text(
                eruption_lon + 3,
                eruption_lat - 6,
                'Eruption\nLocation',
                color='red',
                fontsize=7,
                ha='left',
                va='top',
                transform=map_transform,
                zorder=9,
            )


        arrow_lat = 20
        arrow_x0 = 105+120
        arrow_x1 = 168+120
        map_ax.annotate(
            '',
            xy=(arrow_x1, arrow_lat),
            xytext=(arrow_x0, arrow_lat),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.8),
            transform=map_transform,
            zorder=9,
            alpha=0.2,
        )
        if idx==1:  
            map_ax.text(
                arrow_x0 ,
                arrow_lat + 15.,
                'Plume Drift',
                color='red',
                fontsize=7,
                ha='left',
                va='top',
                transform=map_transform,
                zorder=9,
            )


    # --- 美化---
    for i, ax in enumerate([ax_oh_map, ax_h2o_map]):
        if CARTOPY_AVAILABLE:
            ax.coastlines(color='grey')
            gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
            gl.xlabel_style = {'size': font_settings['ticks']}
            gl.ylabel_style = {'size': font_settings['ticks']}
            gl.top_labels = False
            gl.right_labels = False
            # if i == 0: 
            #     gl.bottom_labels = False
            # add longitude x label for i==1
            if i == 1: 
                gl.bottom_labels = True
                gl.xlabel_style = {'size': font_settings['ticks']}
    ax_h2o_map.set_xlabel('Longitude', fontsize=font_settings['label'])

    # 5. 保存图像
    plot_path = os.path.join(plot_dir, f'Figure4_hunga_tonga_analysis_3col_{source}_{plot_year}_vs_{baseline_year}_day{doy}.pdf')
    plt.savefig(plot_path, dpi=600)
    plt.close(fig)
    print(f"Saved final Hunga-Tonga analysis plot to: {plot_path}")


def plot_profile_comparison(year, doy, plot_dir, OUTPUT_DIR, num_profiles=10, random_seed=42, font_sizes=None):
    """
    绘制OH垂直剖面对比图。MLS观测的不确定度使用误差棒(error bars)形式展示，
    """
    print(f"\n--- Generating OH Profile Comparison for {year}, Day {doy} ---")

    # Apply style first so later rcParams overrides stay effective.
    plt.style.use('seaborn-v0_8-ticks')

    # # --- 1. 设置与数据加载 ---
    # default_font_sizes = {'title': 8, 'label': 9, 'tick': 8, 'legend': 8}
    font_path = 'Plot/fonts/MYRIADPRO-REGULAR.OTF'
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        # 获取加载后的字体真实名称（通常是 'Myriad Pro'）
        prop = fm.FontProperties(fname=font_path)
        myriad_font = prop.get_name()
    else:
        print(f"Warning: Font file {font_path} not found. Using sans-serif.")
        myriad_font = 'Arial'

    # 全局字体配置
    plt.rcParams.update({
        'font.family': [myriad_font],
        'font.sans-serif': [myriad_font],
        'font.cursive': [myriad_font], 
        'font.fantasy': [myriad_font],
        'font.serif': [myriad_font],
        'pdf.fonttype': 42,  
        'mathtext.fontset': 'custom',
        'mathtext.it': f'{myriad_font}:italic',
        'mathtext.rm': myriad_font,
        'mathtext.bf': f'{myriad_font}:bold',
        'mathtext.sf': myriad_font,
        'mathtext.tt': myriad_font,
        'mathtext.cal': myriad_font,
        'axes.unicode_minus': False,
    })

    # 严格按照期刊要求的 6-8 pt 字号
    f_size = {'title': 8, 'label': 8, 'tick': 8, 'legend': 8, 'panel': 9}

    if font_sizes is None: font_sizes = {}
    final_font_sizes = {**f_size, **font_sizes}

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

    # Set the unit to 10^6 cm^-3 for plotting
    for key in loaded_data:
        loaded_data[key] = loaded_data[key] / 1e6

    # --- 3. 绘图 ---
    ncols = 4
    nrows = (num_to_sample + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.25, 5.6), sharey=True)
    plt.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.07,  hspace=0.25)
    axes = axes.flatten() if num_to_sample > 1 else [axes]
    
    plot_styles = {
        'True MLS OH': {'color': 'blue', 'label': 'MLS OH Obs and Uncertainty', 'zorder': 3,'linewidth': 0.8},
        'Predicted MLS-OH': {'color': 'green', 'linestyle': '-','linewidth': 0.8,'marker': '.', 'label': 'DRCAT-Predicted OH', 'zorder': 2}
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
                        linewidth=style['linewidth'],
                        color=style['color'],
                        label=style['label'],
                        zorder=style['zorder'])

        ax.set_yscale('log')
        ax_linewidth = 0.8
        for spine in ax.spines.values():
            spine.set_linewidth(ax_linewidth)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.xaxis.set_tick_params(labelsize=final_font_sizes['tick'])
        ax.yaxis.set_tick_params(labelsize=final_font_sizes['tick'])
        # ax.grid(True, which='major', linestyle='--', alpha=0.7)
        # ax.grid(True, which='minor', linestyle=':', alpha=0.4)
        # ax.tick_params(axis='both', which='major', labelsize=final_font_sizes['tick'])
        # if i >= len(axes) - ncols:
        #     ax.set_xlabel(r'OH Density (10$^{6}$ cm$^{-3}$)', fontsize=final_font_sizes['label'])
    # set text for the bottom center seperately to serve as the x-axis label for the entire figure
    # adjust its location to be centered below the subplots
    # fig.subplots_adjust(bottom=0.12)
    fig.text(0.5, 0.01, r'OH Density (10$^{6}$ cm$^{-3}$)', ha='center', fontsize=final_font_sizes['label'])
    


    # --- 4. 保存 ---
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
                # ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        
        max_p = loaded_data['True MLS OH'].pressure.max().values
        min_p = loaded_data['True MLS OH'].pressure.min().values
        axes[0].set_ylim(max_p*1.05, min_p*0.95)

    handles, labels = [], []
    if num_to_sample > 0:
        first_ax_with_content = next((ax for ax in axes if ax.has_data()), None)
        if first_ax_with_content:
            h, l = first_ax_with_content.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in labels:
                    labels.append(label)
                    handles.append(handle)
    # print(f"Legend handles: {handles}, labels: {labels}")
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(labels), 
                   bbox_to_anchor=(0.5, 1.0), fontsize=final_font_sizes['legend'])

    
    data_date_obj = datetime.strptime(f'{year}-{doy}', '%Y-%j')
    formatted_date = data_date_obj.strftime('%d %b %Y')    
    # fig.text(0.99, 0.98, f'Date: {formatted_date}', ha='right', va='top',
    #          fontsize=11, color='grey', alpha=0.7)
    
    plot_path = os.path.join(plot_dir, f'Figure1_profile_comparison_{year}_day{doy}.pdf')
    plt.savefig(plot_path, dpi=400)
    plt.close(fig)
    print(f"Saved final OH profile comparison plot to: {plot_path}")


def plot_profile_comparison_3line(year, doy, plot_dir, OUTPUT_DIR, num_profiles=10, random_seed=42, font_sizes=None):
    """
    绘制OH垂直剖面对比图。MLS观测的不确定度使用误差棒(error bars)形式展示，
    """
    print(f"\n--- Generating OH Profile Comparison for {year}, Day {doy} ---")

    # Apply style first so later rcParams overrides stay effective.
    plt.style.use('seaborn-v0_8-ticks')

    # # --- 1. 设置与数据加载 ---
    font_path = 'Plot/fonts/MYRIADPRO-REGULAR.OTF'
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        # 获取加载后的字体真实名称（通常是 'Myriad Pro'）
        prop = fm.FontProperties(fname=font_path)
        myriad_font = prop.get_name()
    else:
        print(f"Warning: Font file {font_path} not found. Using sans-serif.")
        myriad_font = 'Arial'

    # 全局字体配置
    plt.rcParams.update({
        'font.family': [myriad_font],
        'font.sans-serif': [myriad_font],
        # 显式重定向 cursive，防止它去系统里乱找
        'font.cursive': [myriad_font], 
        'font.fantasy': [myriad_font],
        'font.serif': [myriad_font],
        'pdf.fonttype': 42,  # 避免导出为轮廓，确保文字可编辑
        'mathtext.fontset': 'custom',
        'mathtext.it': f'{myriad_font}:italic',
        'mathtext.rm': myriad_font,
        'mathtext.bf': f'{myriad_font}:bold',
        'mathtext.sf': myriad_font,
        'mathtext.tt': myriad_font,
        'mathtext.cal': myriad_font,
        'axes.unicode_minus': False,
    })

    # 严格按照期刊要求的 6-8 pt 字号
    f_size = {'title': 8, 'label': 8, 'tick': 8, 'legend': 8, 'panel': 9}

    if font_sizes is None: font_sizes = {}
    final_font_sizes = {**f_size, **font_sizes}

    data_sources = {
        'Predicted MLS-OH': f'predicted_MLS_OH_density_gridded_{year}.nc',
        'Predicted MLS-OH Uncertainty': f'predicted_MLS_OH_uncertainty_density_gridded_{year}.nc',
        'TOMCAT OH': f'TOMCAT_OH_density_gridded_{year}.nc',
        'SSA OH': f'calculated_SSA_OH_density_gridded_{year}.nc'
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

    # Set the unit to 10^6 cm^-3 for plotting
    for key in loaded_data:
        loaded_data[key] = loaded_data[key] / 1e6

    # --- 3. 绘图 ---
    ncols = 4
    nrows = (num_to_sample + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.25, 5.6), sharey=True)
    plt.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.07,  hspace=0.25)
    axes = axes.flatten() if num_to_sample > 1 else [axes]
    
    plot_styles = {
        'True MLS OH': {'color': 'blue', 'label': 'MLS OH Obs and Uncertainty', 'zorder': 3,'linewidth': 0.8},
        'Predicted MLS-OH': {'color': 'green', 'linestyle': '-','linewidth': 0.8,'marker': '.', 'label': 'DRCAT-Predicted OH', 'zorder': 2},
        'TOMCAT OH': {'color': 'orange', 'linestyle': '--','linewidth': 0.8,'marker': 'x', 'label': 'TOMCAT OH', 'zorder': 1},
        'SSA OH': {'color': 'black', 'linestyle': '-.','linewidth': 0.8,'marker': 's', 'markersize': 2, 'label': 'SSA OH', 'zorder': 0}
    }

    for i, point in enumerate(selected_points):
        ax = axes[i]
        lat, lon = point.item()
        lat_str = f"{abs(lat):.1f}°N" if lat >= 0 else f"{abs(lat):.1f}°S"
        lon_str = f"{abs(lon):.1f}°E" if lon >= 0 else f"{abs(lon):.1f}°W"
        ax.set_title(f"{lat_str}, {lon_str}", fontsize=final_font_sizes['title'])

        # --- 绘图逻辑修改 ---
        # 0. PLOT TOMCAT AND SSA PROFILES IF AVAILABLE
        for model_key in ['TOMCAT OH', 'SSA OH']:
            if model_key in loaded_data:
                style = plot_styles[model_key]
                profile = loaded_data[model_key].sel(lat=lat, lon=lon, method='nearest')
                pressure = profile.pressure.values
                ax.plot(profile.values, pressure, **style)



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
                        linewidth=style['linewidth'],
                        color=style['color'],
                        label=style['label'],
                        zorder=style['zorder'])

        ax.set_yscale('log')
        ax_linewidth = 0.8
        for spine in ax.spines.values():
            spine.set_linewidth(ax_linewidth)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.xaxis.set_tick_params(labelsize=final_font_sizes['tick'])
        ax.yaxis.set_tick_params(labelsize=final_font_sizes['tick'])

    fig.text(0.5, 0.01, r'OH Density (10$^{6}$ cm$^{-3}$)', ha='center', fontsize=final_font_sizes['label'])
    


    # --- 4. 最终美化与保存 ---
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
                # ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        
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
    # print(f"Legend handles: {handles}, labels: {labels}")
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=3, 
                   bbox_to_anchor=(0.5, 1.0), fontsize=final_font_sizes['legend'])

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    data_date_obj = datetime.strptime(f'{year}-{doy}', '%Y-%j')
    formatted_date = data_date_obj.strftime('%d %b %Y')    
    # fig.text(0.99, 0.98, f'Date: {formatted_date}', ha='right', va='top',
    #          fontsize=11, color='grey', alpha=0.7)
    
    plot_path = os.path.join(plot_dir, f'Figure1_Supp_profile_comparison_{year}_day{doy}.pdf')
    plt.savefig(plot_path, dpi=400)
    plt.close(fig)
    print(f"Saved final OH profile 3line comparison plot to: {plot_path}")



