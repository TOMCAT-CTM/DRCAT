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


## Usage

```bash
# ETTh1
python -u main_informer.py --model informer --data ETTh1 --attn prob --freq h

# ETTh2
python -u main_informer.py --model informer --data ETTh2 --attn prob --freq h

# ETTm1
python -u main_informer.py --model informer --data ETTm1 --attn prob --freq t
```

## Raw dataset

Raw dataset of MLS L2 Version 5 data can be downloaded from NASA Goddard Space Flight Center Earth Sciences (GES) Data and Information Services Center (DISC): https://disc.gsfc.nasa.gov/

## Contact
For feedback and questions, please contact us at windosryin@whu.edu.cn