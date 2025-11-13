# DRCAT

:triangular_flag_on_post:(Nov 13, 2025): This paper is currently under submission, therefore this repo is currently under preparation. 

This the origin implementation of DRCAT in the following paper:
A Chemistry-informed Deep Learning Networks for Mitigating Stratospheric OH Data Gap 

This repository provide example code to run DRCAT model. It also provides pretrained model weights, normalization statistics and input data.
<p align="center">
<img src=".\img\model.png" height = "360" alt="" align=center />
<br><br>
<b>Figure 1.</b> The architecture of DRCAT.
</p>


## Brief description of relevant library files

*   `run.py`: Wrapper used to run the inference of DRCAT
    to produce a sequence of predictions and save as netcdf files

*   `model.py`: Wrapper used to run the inference of DRCAT
    to produce a sequence of predictions and save as netcdf files

*   `Plot/Plot_function.py`: definition of plotting functions used in this paper

*   `Plot/plot_results.py`: Wrapper used to run the plotting code




## Usage

```bash
# single gpu
python -u run.py 

# multi-gpu
python -u run.py
```


## Dataset
Training data and inference data can be downloaded from google drive: 

Please extract the zip file into the Data directory.

Raw dataset of MLS L2 Version 5 data can be downloaded from NASA Goddard Space Flight Center Earth Sciences (GES) Data and Information Services Center (DISC): https://disc.gsfc.nasa.gov/

## Contact
For feedback and questions, please contact us at windosryin@whu.edu.cn