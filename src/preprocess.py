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
    train = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv.gz'))
    test = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv.gz'))

    base_path = os.path.join(DATA_FOLDER, 'base_data.pkl')
    if os.path.exists(base_path):
        # pickleで前処理したものを保存してる
        data = pd.read_pickle(base_path)
    else:
        data = preprocess(train, test)
        data.to_pickle(base_path)
        print('Finish preprocess')
    
    return data