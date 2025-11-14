# train.py
# -*- coding: utf-8 -*-
"""
最终版代码：基于解耦模块和预测头注入策略的OH预测模型训练脚本。
(V3: 已增加DDP支持，用于多GPU分布式训练)

该脚本严格遵循以下最优工作流程：
1.  训练一个独立的“物理信息模块”，使其能根据化学输入精准预测无噪声的SSA-OH。
2.  使用训练好的物理模块，为具有真实MLS观测的数据点生成“predicted_SSA_OH”作为一项新的物理基准特征。
3.  训练一个独立的“MLS预测模块”，其编码器权重由物理模块迁移而来。
    该模块接收原始化学特征，并通过其编码器生成化学环境表征；
    然后，在最终的预测头，将此化学表征与“predicted_SSA_OH”物理基准拼接，共同预测带有仪器噪声的MLS-OH。
"""
import math
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.metrics
import joblib
import json
import argparse
from model import DRCAT
from inference import *
# +++++++++++++++++++++++++++++++++++++

DATA_PATH = '/scratch/pdpv7239/SSA_dataset/'
BASE_PATH = '/scratch/pdpv7239/SSA_dataset_base/'
PLOT_OUTPUT_DIR = 'Data/DRCAT_V10'
CHEM_FEATURES = [ 'H2O', 'O3', 'Temperature', 'HNO3','N2O','HCl','HO2']

pressure_levels = np.load('Data/DRCAT_V10/pressure_levels.npy')

for INFERENCE_YEAR in range(2021,2023):
    INFERENCE_DATA_PATH = f'/scratch/pdpv7239/SSA_dataset_base/ml_ready_OH_data_FE_augmented_{INFERENCE_YEAR}.parquet'
    if os.path.exists(INFERENCE_DATA_PATH):
        run_inference(
            inference_data_path=INFERENCE_DATA_PATH,
            chem_features=CHEM_FEATURES,
            p_levels=pressure_levels,
            plot_dir=PLOT_OUTPUT_DIR
        )
