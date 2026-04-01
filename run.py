# train.py
# -*- coding: utf-8 -*-
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
pressure_levels = np.load('Data/DRCAT_V10/pressure_levels.npy')

PLOT_OUTPUT_DIR = 'Data/DRCAT_V10'
CHEM_FEATURES = [ 'H2O', 'O3', 'Temperature', 'HNO3','N2O','HCl','HO2']


for INFERENCE_YEAR in [2005,2006,2007,2008,2009,2011,2014,2021,2022]: #range(2011,2012):
    INFERENCE_DATA_PATH = f'Data/base_data/ml_ready_OH_data_FE_augmented_{INFERENCE_YEAR}.parquet'
    if os.path.exists(INFERENCE_DATA_PATH):
        run_inference(
            inference_data_path=INFERENCE_DATA_PATH,
            chem_features=CHEM_FEATURES,
            p_levels=pressure_levels,
            plot_dir=PLOT_OUTPUT_DIR
        )
