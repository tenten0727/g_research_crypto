# train, testを統合した基本的な前処理をしたデータ作成

import pandas as pd
import os
import numpy as np
from itertools import product
from pandas.core.indexes import base
from sklearn.preprocessing import LabelEncoder

DATA_FOLDER = '../input'

def preprocess(train, test):
    matrix = []
    return matrix

def base_data():
    print('++++++ Data Loading ++++++')
    train_pkl_path = os.path.join(DATA_FOLDER, 'train.pkl')
    if os.path.exists(train_pkl_path):
        data = pd.read_pickle(train_pkl_path)
    else:
        data = pd.read_csv(os.path.join(DATA_FOLDER, 'g-research-crypto-forecasting', 'train.csv'))
        # kaggleでそのまま使えるようにするため
        if os.access(train_pkl_path, os.W_OK):
            data.to_pickle(train_pkl_path)

    asset_detail = pd.read_csv(os.path.join(DATA_FOLDER, 'g-research-crypto-forecasting', 'asset_details.csv'))
    
    data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')

    return data, asset_detail