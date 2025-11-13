import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from Plot_function import *

OUTPUT_DIR = '/scratch/pdpv7239/experiment1026_v10/DRCAT_test_0'#'/scratch/pdpv7239/PINN_V1_test1_tomcat'  # This should match the output dir in PINN.py
PLOT_OUTPUT_DIR = os.path.join('Plot/Results/experiment_1026_v10/DRCAT_test_0') # Directory for the new plots
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)    


if __name__ == "__main__":

    #########################
    for doy in range(1, 100, 10):  
        plot_profile_comparison(
        year=2008,
        doy=doy,
        plot_dir=PLOT_OUTPUT_DIR,      
        OUTPUT_DIR=OUTPUT_DIR,    
        num_profiles=8            # 绘制剖面数量
    )



    plot_zonal_mean_comparison_sci(year=2006, day_of_year=100, plot_dir=PLOT_OUTPUT_DIR, output_dir=OUTPUT_DIR)
    plot_zonal_mean_comparison_sci(year=2007, day_of_year=100, plot_dir=PLOT_OUTPUT_DIR, output_dir=OUTPUT_DIR)
    plot_zonal_mean_comparison_sci(year=2008, day_of_year=100, plot_dir=PLOT_OUTPUT_DIR, output_dir=OUTPUT_DIR)
    plot_zonal_mean_comparison_sci(year=2009, day_of_year=100, plot_dir=PLOT_OUTPUT_DIR, output_dir=OUTPUT_DIR)
    plot_zonal_mean_comparison_sci(year=2006, day_of_year=240, plot_dir=PLOT_OUTPUT_DIR, output_dir=OUTPUT_DIR)
    plot_zonal_mean_comparison_sci(year=2007, day_of_year=240, plot_dir=PLOT_OUTPUT_DIR, output_dir=OUTPUT_DIR)
    plot_zonal_mean_comparison_sci(year=2008, day_of_year=240, plot_dir=PLOT_OUTPUT_DIR, output_dir=OUTPUT_DIR)
    plot_zonal_mean_comparison_sci(year=2009, day_of_year=240, plot_dir=PLOT_OUTPUT_DIR, output_dir=OUTPUT_DIR)

    for year_extend in range(2011, 2015):  # 每年绘制一次
        for doy in range(220, 250, 2):  # 每10天绘制一次
            plot_zonal_mean_comparison_sci(year=year_extend, day_of_year=doy, plot_dir=PLOT_OUTPUT_DIR, output_dir=OUTPUT_DIR)

 
    ############################################
    # # # 
    for day in range(34, 35, 1):  # 每天绘制一次
        plot_hunga_tonga_analysis(

            plot_year=2022,
            baseline_year=2021,
            doy=day,  
            plot_dir=PLOT_OUTPUT_DIR, 
            OUTPUT_DIR=OUTPUT_DIR, 
            pressure_level=22
        )
    for day in range(35, 100, 5):  # 每5天绘制一次
        plot_hunga_tonga_analysis(

            plot_year=2022,
            baseline_year=2021,
            doy=day, 
            plot_dir=PLOT_OUTPUT_DIR,  
            OUTPUT_DIR=OUTPUT_DIR, 
            pressure_level=22
        )

  
    ############################################

    # 运行每日误差分析和绘图 
    analysis_years = range(2007, 2010)
    plot_comprehensive_error_analysis(analysis_years, OUTPUT_DIR, PLOT_OUTPUT_DIR)
    
    # # 运行其他绘图函数
    # ALL_YEARS = range(2005, 2023)
    # pressure_levels = np.load('pressure_levels_OH2.npy')
    # plot_temporal_trends(ALL_YEARS, pressure_levels, OUTPUT_DIR, PLOT_OUTPUT_DIR) # 选择几个压力水平来绘图

    ######################################################
    plot_args = {
        'fig_h': 3,
        'fig_w': 10,
        'cmap': 'jet'  
    }


    plot_zonal_mean_timeline(
        variable_name='Calculated SSA-OH',
        start_day=10,
        end_day=100,
        step_days=10,
        plot_dir=PLOT_OUTPUT_DIR,
        OUTPUT_DIR=OUTPUT_DIR,
        year=2022,
        plot_args=plot_args

    )

    plot_zonal_mean_timeline(
        variable_name='TOMCAT OH',
        start_day=10,
        end_day=100,
        step_days=10,
        plot_dir=PLOT_OUTPUT_DIR,
        OUTPUT_DIR=OUTPUT_DIR,
        year=2022,
        plot_args=plot_args

    )
    
    plot_args = {
        'fig_h': 3,
        'fig_w': 10,
        'cmap': 'RdBu_r' 
    }
    # 示例：绘制ML-Predicted OH的异常值时间序列图
    plot_zonal_mean_anomaly_timeline(
        variable_name='ML-Predicted OH',
        start_day=10,
        end_day=100,
        step_days=10,
        plot_dir=PLOT_OUTPUT_DIR,
        OUTPUT_DIR=OUTPUT_DIR,
        baseline_year=2021,
        target_year=2022,
        plot_args=plot_args
    )

    plot_zonal_mean_anomaly_timeline(
        variable_name='Calculated SSA-OH',
        start_day=10,
        end_day=100,
        step_days=10,
        plot_dir=PLOT_OUTPUT_DIR,
        OUTPUT_DIR=OUTPUT_DIR,
        baseline_year=2021,
        target_year=2022,
        plot_args=plot_args
    )