# inference.py
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import joblib
from tqdm import tqdm
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from matplotlib.colors import LogNorm

# 导入模型架构
from model import DRCAT

def _load_model_config(plot_dir):
   
    config_path = os.path.join(plot_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}. ")
                               
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 提取模型超参数
    model_params = {
        "emb_size": config.get("EMB_SIZE", 128),
        "nhead": config.get("N_HEAD", 8),
        "num_layers": config.get("N_LAYERS", 3),
        "dim_feedforward": config.get("DIM_FEEDFORWARD", 512),
        "dropout": config.get("DROPOUT", 0.1),
        "gnn_dim": config.get("GNN_DIM", 64),
    }
    return model_params


def reshape_and_filter_data(df, features, targets, n_levels):
    """
    清理数据，保留完整的垂直剖面，并重塑为(样本数, 层数, 特征数)的格式。
    """
    df['unique_profile_id'] = df.groupby(['Time', 'lat', 'lon']).ngroup()
    profile_counts = df.groupby('unique_profile_id')['pressure'].count()
    complete_profile_ids = profile_counts[profile_counts == n_levels].index
    df_complete = df[df['unique_profile_id'].isin(complete_profile_ids)]
    
    profile_data = df_complete.sort_values(['unique_profile_id', 'pressure'])
    n_profiles = df_complete['unique_profile_id'].nunique()
    
    X_raw = profile_data[features].values.reshape(n_profiles, n_levels, len(features))
    Y = {target: profile_data[target].values.reshape(n_profiles, n_levels) for target in targets}
    
    mask_bool = np.isnan(X_raw)
    mask = 1.0 - mask_bool.astype(float)
    X = np.nan_to_num(X_raw, nan=0.0)
    
    return X, mask, Y, n_profiles, profile_data

@torch.no_grad()
def generate_features_with_chemical_module(df_in, chem_features, p_levels, model_path, scaler_path, device, batch_size=1024):
    n_levels = len(p_levels)
    n_chem_features = len(chem_features)
    
    plot_dir = os.path.dirname(model_path)
    model_params = _load_model_config(plot_dir)
    
    model = DRCAT(
        n_chem_features, n_levels, k=n_levels-1, **model_params
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=f'cuda:{device}'))
    model.eval()
    
    x_scaler, y_scaler_phys = joblib.load(scaler_path)
    
    X, M, _, n_profiles, df_complete  = reshape_and_filter_data(df_in, chem_features, [], n_levels)
    X_scaled = x_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    
    loader = DataLoader(TensorDataset(torch.tensor(X_scaled), torch.tensor(M)), batch_size=batch_size, shuffle=False)
    
    all_preds_scaled = []
    
    for x_b, m_b in tqdm(loader, desc="Generating Features"):
        x_b, m_b = x_b.to(device, dtype=torch.float32), m_b.to(device, dtype=torch.float32)
        preds_scaled_batch = model(x_b, mask=m_b)
        all_preds_scaled.append(preds_scaled_batch.cpu().numpy())
    
    preds_scaled = np.concatenate(all_preds_scaled)
    preds_orig = y_scaler_phys.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(n_profiles, n_levels)
    
    df_out = df_complete.copy()
    df_out['predicted_SSA_OH'] = preds_orig.flatten()
    
    return df_out


# --- 2. 独立推理与分析工作流 ---

def vmr_to_number_density(vmr, pressure_pa, temperature_k):
    """
    将体积混合比 (VMR) 转换为数密度 (molecules/cm^3)。
    """
    k_boltzmann = 1.380649e-23  # J/K
    vmr = np.asarray(vmr)
    pressure_pa = np.asarray(pressure_pa)
    temperature_k = np.asarray(temperature_k)
    
    valid_temp_mask = temperature_k > 0
    number_density_cm3 = np.full(vmr.shape, np.nan)
    
    n_air_valid = pressure_pa[valid_temp_mask] / (k_boltzmann * temperature_k[valid_temp_mask])
    number_density_m3 = vmr[valid_temp_mask] * n_air_valid
    number_density_cm3[valid_temp_mask] = number_density_m3 * 1e-6
    
    return number_density_cm3

def create_gridded_netcdf(df_with_preds, var_name, p_levels, output_dir, lat_res=2.78, lon_res=2.8125):

    print(f" Create NC file for '{var_name}' ...")
    if var_name not in df_with_preds.columns or df_with_preds[var_name].isnull().all():
        print(f"  '{var_name}' skipped")
        return

    year = df_with_preds['Year'].iloc[0]
    days_in_year = 366 if pd.to_datetime(f'{year}-12-31').dayofyear == 366 else 365
    lats_grid = np.arange(-90, 90 + lat_res, lat_res)
    lons_grid = np.arange(0, 360, lon_res)
    doy_grid = np.arange(1, days_in_year + 1)

    df_grid = df_with_preds.copy()
    df_grid['doy'] = pd.to_datetime(df_grid['Time']).dt.dayofyear
    df_grid['lat_idx'] = pd.cut(df_grid['lat'], bins=lats_grid, labels=False, right=False)
    df_grid['lon_idx'] = pd.cut(df_grid['lon'], bins=lons_grid, labels=False, right=False)
    
    aggregated_data = df_grid.groupby(['doy', 'pressure', 'lat_idx', 'lon_idx'])[var_name].mean()

    output_array = np.full((len(doy_grid), len(p_levels), len(lats_grid)-1, len(lons_grid)-1), np.nan)
    
    doy_indices_raw = aggregated_data.index.get_level_values('doy').to_numpy() - 1
    pressure_map = {level: i for i, level in enumerate(p_levels)}
    pressure_indices_float = aggregated_data.index.get_level_values('pressure').map(pressure_map).to_numpy()
    
    lat_indices_raw = aggregated_data.index.get_level_values('lat_idx').to_numpy()
    lon_indices_raw = aggregated_data.index.get_level_values('lon_idx').to_numpy()
    values_raw = aggregated_data.to_numpy()

    valid_mask = ~np.isnan(pressure_indices_float)
    
    doy_indices = doy_indices_raw[valid_mask].astype(int)
    pressure_indices = pressure_indices_float[valid_mask].astype(int)
    lat_indices = lat_indices_raw[valid_mask].astype(int)
    lon_indices = lon_indices_raw[valid_mask].astype(int)
    values = values_raw[valid_mask]
    
    valid_idx_mask = (doy_indices >= 0) & (doy_indices < len(doy_grid)) & \
                     (pressure_indices >= 0) & (pressure_indices < len(p_levels)) & \
                     (lat_indices >= 0) & (lat_indices < len(lats_grid)-1) & \
                     (lon_indices >= 0) & (lon_indices < len(lons_grid)-1)

    output_array[doy_indices[valid_idx_mask], 
                 pressure_indices[valid_idx_mask], 
                 lat_indices[valid_idx_mask], 
                 lon_indices[valid_idx_mask]] = values[valid_idx_mask]

    da = xr.DataArray(
        data=output_array,
        dims=['doy', 'pressure', 'lat', 'lon'],
        coords={
            'doy': doy_grid,
            'pressure': p_levels,
            'lat': lats_grid[:-1] + lat_res / 2, 
            'lon': lons_grid[:-1] + lon_res / 2
        },
        name=var_name
    )
    da.attrs['long_name'] = f'Predicted {var_name.replace("_density", "")} Number Density'
    da.attrs['units'] = 'molecules cm-3'; da.pressure.attrs['units'] = 'hPa'
    output_path = os.path.join(output_dir, f'{var_name}_gridded_{year}.nc')
    da.to_netcdf(output_path)
    print(f"saved at : {output_path}")

def plot_inference_anomaly(day_of_year, target_year, baseline_year, plot_dir):
    """
    加载两个年份的NetCDF推理结果，计算并绘制指定日的相对变化（异常百分比）。
    """
    try:
        ds_ssa_target = xr.open_dataset(os.path.join(plot_dir, f'predicted_SSA_OH_density_gridded_{target_year}.nc'))
        ds_mls_target = xr.open_dataset(os.path.join(plot_dir, f'predicted_MLS_OH_density_gridded_{target_year}.nc'))
        ds_ssa_baseline = xr.open_dataset(os.path.join(plot_dir, f'predicted_SSA_OH_density_gridded_{baseline_year}.nc'))
        ds_mls_baseline = xr.open_dataset(os.path.join(plot_dir, f'predicted_MLS_OH_density_gridded_{baseline_year}.nc'))
    except FileNotFoundError as e:
        print(f"错误: 找不到NetCDF文件。请确保已为 {target_year} 和 {baseline_year} 年都成功运行了推理模块。 Error: {e}")
        return

    ssa_zonal_target = ds_ssa_target[list(ds_ssa_target.data_vars)[0]].sel(doy=day_of_year, method="nearest").mean(dim='lon', skipna=True)
    mls_zonal_target = ds_mls_target[list(ds_mls_target.data_vars)[0]].sel(doy=day_of_year, method="nearest").mean(dim='lon', skipna=True)
    ssa_zonal_baseline = ds_ssa_baseline[list(ds_ssa_baseline.data_vars)[0]].sel(doy=day_of_year, method="nearest").mean(dim='lon', skipna=True)
    mls_zonal_baseline = ds_mls_baseline[list(ds_mls_baseline.data_vars)[0]].sel(doy=day_of_year, method="nearest").mean(dim='lon', skipna=True)

    epsilon = 1e-12
    ssa_anomaly = ((ssa_zonal_target - ssa_zonal_baseline) / (ssa_zonal_baseline + epsilon)) * 100
    mls_anomaly = ((mls_zonal_target - mls_zonal_baseline) / (mls_zonal_baseline + epsilon)) * 100

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), sharex=True, sharey=True)
    fig.suptitle(f'Zonal Mean OH Anomaly for Day {day_of_year} ({target_year} vs {baseline_year})', fontsize=16)
    
    anomaly_cmap = 'RdBu_r'
    max_anomaly = 100

    ssa_anomaly.plot.pcolormesh(
        ax=axes[0], x='lat', y='pressure', cmap=anomaly_cmap, vmin=-max_anomaly, vmax=max_anomaly,
        cbar_kwargs={'label': 'OH Change (%)', 'extend': 'both'}
    )
    axes[0].invert_yaxis(); axes[0].set_title('Predicted SSA-OH Anomaly'); axes[0].set_xlabel('')

    mls_anomaly.plot.pcolormesh(
        ax=axes[1], x='lat', y='pressure', cmap=anomaly_cmap, vmin=-max_anomaly, vmax=max_anomaly,
        cbar_kwargs={'label': 'OH Change (%)', 'extend': 'both'}
    )
    axes[1].invert_yaxis(); axes[1].set_title('Predicted MLS-OH Anomaly (Final Model)'); axes[1].set_xlabel('Latitude')

    for ax in axes:
        ax.set_yscale('log'); ax.set_ylabel('Pressure (hPa)'); ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_filename = os.path.join(plot_dir, f'inference_anomaly_day_{day_of_year}_{target_year}_vs_{baseline_year}.png')
    plt.savefig(output_filename, dpi=300)
    print(f"✅ 相对变化图已保存为: {output_filename}")
    plt.close(fig) # 关闭图形


@torch.no_grad()
def run_inference(inference_data_path, chem_features, p_levels, plot_dir, device=0, batch_size=1024):

    print("\n" + "#"*80)
    print(f"# inference start: {inference_data_path}")
    print("#"*80)

    try:
        df_inference = pd.read_parquet(inference_data_path)
        df_inference.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_inference = df_inference[df_inference['Source'] == 'Base'].copy()
    except Exception as e:
        print(f"Error: {e}"); return
    
    n_levels = len(p_levels); n_chem_features = len(chem_features)
    
    # 加载模型配置
    model_params = _load_model_config(plot_dir)
    
    phys_model_path = os.path.join(plot_dir, 'physics_module.pth'); phys_scalers_path = os.path.join(plot_dir, 'scalers_physics.pkl')
    phys_model = DRCAT(n_chem_features, n_levels, k=n_levels-1, **model_params).to(device)
    phys_model.load_state_dict(torch.load(phys_model_path, map_location=f'cuda:{device}')); phys_model.eval()
    x_scaler_phys, y_scaler_phys = joblib.load(phys_scalers_path)
    
    mls_model_path = os.path.join(plot_dir, 'mls_prediction_module.pth'); mls_scalers_path = os.path.join(plot_dir, 'scalers_mls.pkl')
    mls_model = DRCAT(n_chem_features, n_levels, k=n_levels-1, **model_params).to(device)
    mls_model.load_state_dict(torch.load(mls_model_path, map_location=f'cuda:{device}')); mls_model.eval()
    x_scaler_mls, y_scaler_mls, z_scaler_mls = joblib.load(mls_scalers_path)

    X_data, M_data, _, n_profiles, df_filtered = reshape_and_filter_data(df_inference, chem_features, [], n_levels)
    X_scaled_phys = x_scaler_phys.transform(X_data.reshape(-1, n_chem_features)).reshape(X_data.shape)
    loader_phys = DataLoader(TensorDataset(torch.tensor(X_scaled_phys, dtype=torch.float32), torch.tensor(M_data, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    
    ssa_preds_scaled = torch.cat([phys_model(x_b.to(device), mask=m_b.to(device)).cpu() for x_b, m_b in tqdm(loader_phys, "Predicting SSA-OH")])
    predicted_ssa_oh_vmr = y_scaler_phys.inverse_transform(ssa_preds_scaled.numpy().reshape(-1, 1)).reshape(n_profiles, n_levels)

    X_scaled_mls = x_scaler_mls.transform(X_data.reshape(-1, n_chem_features)).reshape(X_data.shape)
    Z_ssa_scaled = z_scaler_mls.transform(predicted_ssa_oh_vmr.reshape(-1, 1)).reshape(n_profiles, n_levels, 1)
    loader_mls = DataLoader(TensorDataset(torch.tensor(X_scaled_mls), torch.tensor(M_data), torch.tensor(Z_ssa_scaled)), batch_size=batch_size, shuffle=False)
    
    mls_preds_scaled, mls_unc_scaled = [], []
    for x_b, m_b, z_b in tqdm(loader_mls, desc="Predicting MLS-OH"):
        mean, logvar = mls_model(x_b.to(device, dtype=torch.float32), mask=m_b.to(device, dtype=torch.float32), ssa_phys_pred=z_b.to(device, dtype=torch.float32))
        mls_preds_scaled.append(mean.cpu())
        mls_unc_scaled.append(logvar.cpu())
    
    predicted_mls_oh_vmr = y_scaler_mls.inverse_transform(torch.cat(mls_preds_scaled).numpy().reshape(-1, 1)).reshape(n_profiles, n_levels)
    predicted_mls_oh_unc_vmr = (np.exp(0.5 * torch.cat(mls_unc_scaled).numpy().reshape(-1, 1)) * y_scaler_mls.scale_).reshape(n_profiles, n_levels)

    results_df = df_filtered.copy()
    results_df['predicted_SSA_OH_vmr'] = predicted_ssa_oh_vmr.flatten()
    results_df['predicted_MLS_OH_vmr'] = predicted_mls_oh_vmr.flatten()
    results_df['predicted_MLS_OH_uncertainty_vmr'] = predicted_mls_oh_unc_vmr.flatten()
    
    pressure_pa = results_df['pressure'].values * 100
    temperature_k = results_df['Temperature'].values

    vars_to_convert = {
        'predicted_SSA_OH': results_df['predicted_SSA_OH_vmr'],
        'predicted_MLS_OH': results_df['predicted_MLS_OH_vmr'],
        'predicted_MLS_OH_uncertainty': results_df['predicted_MLS_OH_uncertainty_vmr'],
        'calculated_SSA_OH': results_df.get('OH'), 
        'true_MLS_OH': results_df.get('MLS_OH'),
        'TOMCAT_OH': results_df.get('TOMCAT_OH'),
        'H2O': results_df.get('H2O')
    }
    
    for name, data in vars_to_convert.items():
        if data is not None:
            results_df[f'{name}_density'] = vmr_to_number_density(data, pressure_pa, temperature_k)

    vars_to_save = [
        'predicted_MLS_OH_density',
        'predicted_MLS_OH_uncertainty_density', 'calculated_SSA_OH_density',
        'true_MLS_OH_density', 'TOMCAT_OH_density', 'H2O'
    ]  #'predicted_SSA_OH_density',
    
    for var_name in vars_to_save:
        create_gridded_netcdf(results_df, var_name, p_levels, plot_dir)
    
    print(f"---inference finish---")


    