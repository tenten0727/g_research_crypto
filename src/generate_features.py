import pandas as pd
import numpy as np
from itertools import groupby, product
import os
from scipy.sparse.sputils import upcast
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer
from sklearn.utils.sparsefuncs import inplace_column_scale
from feature_base import Feature, get_arguments, generate_features, load_datasets
from preprocess import base_data
from sklearn.model_selection import KFold
import glob
from sklearn.cluster import KMeans

Feature.dir = '../features'
data, asset_detail = base_data()

def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])

def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']

class Base(Feature):
    def create_features(self):
        self.data = data
        self.create_memo('Base feature')

class Shadow_features(Feature):
    def create_features(self):
        df_data = data.copy()
        df_data['Upper_Shadow'] = upper_shadow(df_data)
        df_data['Lower_Shadow'] = lower_shadow(df_data)

        self.data = df_data[['Upper_Shadow', 'Lower_Shadow']]
        self.create_memo('Shadow features')

def run():
    if not os.path.isdir(Feature.dir):
        os.makedirs(Feature.dir)
    if not os.path.isfile(Feature.dir+'/_feature_memo.csv'):
        with open(Feature.dir+'/_feature_memo.csv', 'w') as f:
            f.write('特徴量名,説明,カラム\n')
    
    args = get_arguments()
    generate_features(globals(), args.force)

if __name__ == '__main__':
    # 以下はグローバル変数になる。変数のエラー注意。
    run()
