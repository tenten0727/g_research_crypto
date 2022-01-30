# train, testを統合した基本的な前処理をしたデータ作成

import pandas as pd
import os
import numpy as np
from itertools import product
from pandas.core.indexes import base
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
    scales = []
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(data.mean())

    for asset_id in data['Asset_ID'].unique():
        asset_data = data[data['Asset_ID']==asset_id]
        ss = StandardScaler()
        scale_data = ss.fit_transform(asset_data[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']])
        data.loc[data['Asset_ID']==asset_id, ['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']] = scale_data
    
    print(data.head())
    return data, asset_detail