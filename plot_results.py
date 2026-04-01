import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from Plot.Plot_function import *

OUTPUT_DIR = 'Data/DRCAT_V10' # Directory where the processed data files are stored
PLOT_OUTPUT_DIR = os.path.join('Plot/Results/') # Directory for the new plots
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)    


if __name__ == "__main__":

    #########################

    # ## figure 1: profile comparison for 23 March 2006 (DOY 82)
    for doy in range(81, 82):  
        plot_profile_comparison(
        year=2006,
        doy=doy,
        plot_dir=PLOT_OUTPUT_DIR,      
        OUTPUT_DIR=OUTPUT_DIR,    
        num_profiles=8            
    )
        
    for doy in range(81, 82):  
        plot_profile_comparison_3line(
        year=2006,
        doy=doy,
        plot_dir=PLOT_OUTPUT_DIR,      
        OUTPUT_DIR=OUTPUT_DIR,    
        num_profiles=8            
    )

    ## figure 2: daily zonal mean comparison for 10 April,2007

    plot_zonal_mean_comparison_sci(year=2007, day_of_year=100, plot_dir=PLOT_OUTPUT_DIR, output_dir=OUTPUT_DIR)

    # # figure 2-supplement: daily zonal mean difference comparison for 10 April,2007
    plot_zonal_mean_difference_sci(year=2007, day_of_year=100, plot_dir=PLOT_OUTPUT_DIR, output_dir=OUTPUT_DIR)

    # figure Supplement
    plot_zonal_mean_comparison_sci(year=2011, day_of_year=228, plot_dir=PLOT_OUTPUT_DIR, output_dir=OUTPUT_DIR)
    plot_zonal_mean_comparison_sci(year=2014, day_of_year=248, plot_dir=PLOT_OUTPUT_DIR, output_dir=OUTPUT_DIR)


    # figure s9
    plot_args = {
        'row_h': 1.5,
        'cmap_mean': 'jet',
        'cmap_anomaly': 'RdBu_r',
        'anomaly_abs_max': 100,
    }

    start_day = 10
    end_day = 80
    step_days = 15
    plot_zonal_mean_and_anomaly_timeline(
        variable_name='Calculated SSA-OH',
        start_day=start_day,
        end_day=end_day,
        step_days=step_days,
        plot_dir=PLOT_OUTPUT_DIR,
        OUTPUT_DIR=OUTPUT_DIR,
        baseline_year=2021,
        target_year=2022,
        plot_args=plot_args
    )


    # figure 3: Statistical metrics for zonal mean comparison 
    # 运行每日误差分析和绘图 
    analysis_years = range(2007, 2010)
    plot_comprehensive_error_analysis(analysis_years, OUTPUT_DIR, PLOT_OUTPUT_DIR)
   
    # # figure 4: hunga analysis
    plot_hunga_tonga_analysis_log(
            source='DRCAT-Predicted OH',
            plot_year=2022,
            baseline_year=2021,
            doy=25,  
            plot_dir=PLOT_OUTPUT_DIR, 
            OUTPUT_DIR=OUTPUT_DIR, 
            pressure_level=22
        )
    plot_hunga_tonga_analysis_log(
            source='SSA OH',
            plot_year=2022,
            baseline_year=2021,
            doy=25,  
            plot_dir=PLOT_OUTPUT_DIR, 
            OUTPUT_DIR=OUTPUT_DIR, 
            pressure_level=22
        )
    # figure S5

    plot_hunga_tonga_analysis_log(
            source='DRCAT-Predicted OH',
            plot_year=2022,
            baseline_year=2021,
            doy=34,  
            plot_dir=PLOT_OUTPUT_DIR, 
            OUTPUT_DIR=OUTPUT_DIR, 
            pressure_level=22
        )

    plot_hunga_tonga_analysis_log(
            source='SSA OH',
            plot_year=2022,
            baseline_year=2021,
            doy=34,  
            plot_dir=PLOT_OUTPUT_DIR, 
            OUTPUT_DIR=OUTPUT_DIR, 
            pressure_level=22
        )


