# train, testを統合した基本的な前処理をしたデータ作成

import pandas as pd
import os
import numpy as np
from itertools import product
from pandas.core.indexes import base
from sklearn.preprocessing import LabelEncoder

DATA_FOLDER = '../data'

def preprocess(train, test):
    matrix = []
    return matrix

def base_data():
    print('++++++ Data Loading ++++++')
    data = pd.read_pickle(os.path.join(DATA_FOLDER, 'train.pkl'))
    asset_detail = pd.read_csv(os.path.join(DATA_FOLDER, 'asset_details.csv'))
    
    data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')

    return data, asset_detail