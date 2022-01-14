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
import talib

import sys
sys.path.append('../src')
from utils import RunningMean

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
        self.data['Upper_Shadow'] = upper_shadow(data)
        self.data['Lower_Shadow'] = lower_shadow(data)

        self.create_memo('Shadow features')

class Arithmetic_operations(Feature):
    def create_features(self):
        self.data['high_low_div'] = data['High'] / data['Low']
        self.data['open_close_div'] = data['Open'] / data['Close']

        self.data["open_close_sub"] = data["Open"] - data["Close"]
        self.data["high_low_sub"] = data["High"] - data["Low"]

        self.data['hlco_ration'] = self.data["open_close_sub"] / self.data["high_low_sub"]

        self.create_memo('四則演算で計算した特徴量')

# https://www.kaggle.com/yamqwe/crypto-prediction-technical-analysis-features
class Window_feature(Feature):
    def create_features(self):
        self.data['Asset_ID'] = data['Asset_ID']
        asset_group_close = data.groupby('Asset_ID').Close
        l_window = [5, 12, 26, 60]
        for i in l_window:
            self.data['moving_average_'+str(i)] = asset_group_close.transform(lambda x: x.rolling(window=i).mean())
            self.data['moving_std_'+str(i)] = asset_group_close.transform(lambda x: x.rolling(window=i).std())
            
            # 指数移動平均
            # self.data['exponential_moving_average_'+str(i)] = asset_group_close.transform(lambda x: x.ewm(min_periods=i, span=i).mean())
            
            # volumeの移動平均
            self.data['volume_moving_average_'+str(i)] = data.groupby('Asset_ID').Volume.transform(lambda x: x.rolling(window=i).mean())
            
            # Bollinger Band
            self.data['bollinger_band_high_'+str(i)] = self.data['moving_average_'+str(i)] + 2 * self.data['moving_std_'+str(i)]
            self.data['bollinger_band_low_'+str(i)] = self.data['moving_average_'+str(i)] - 2 * self.data['moving_std_'+str(i)]

            # 相対力指数（RSI）...相場の過熱感を一定期間の終値から計算するオシレータ系指標
            self.data['RSI_'+str(i)] = asset_group_close.transform(lambda x: talib.RSI(x.values, i))

            # ２重指数移動平均(DEMA: Double Exponential Moving Average)
            self.data['double_ema_'+str(i)] = asset_group_close.transform(lambda x: talib.DEMA(x, timeperiod=i))

            # Kaufmanの適応型移動平均(KAMA: Kaufman Adaptive Moving Average)
            self.data['kama_'+str(i)] = asset_group_close.transform(lambda x: talib.KAMA(x, timeperiod=i))

        self.data['close_div_ma_60'] = data['Close'] / self.data['moving_average_60']

        self.data.drop('Asset_ID', axis=1, inplace=True)
        self.create_memo('テクニカル分析の際に用いる指標に関する特徴量')

# Ta-Libを利用して算出
class Trend_Line(Feature):
    def create_features(self):
        asset_group_close = data.groupby('Asset_ID').Close

        # トレンドライン(Hilbert Transform - Instantaneous Trendline)
        self.data['ht_trendline'] = asset_group_close.transform(lambda x: talib.HT_TRENDLINE(x))

        self.create_memo('trend_line特徴量')

class MACD(Feature):
    def create_features(self):
        self.data['Asset_ID'] = data['Asset_ID']

        for i in data.Asset_ID.unique():
            self.data.loc[self.data.Asset_ID==i, 'MACD'], self.data.loc[self.data.Asset_ID==i, 'MACD_signal'], self.data.loc[self.data.Asset_ID==i, 'MACD_hist'] = talib.MACD(data[data.Asset_ID==i].Close, fastperiod=12, slowperiod=26, signalperiod=9)

        self.data.drop('Asset_ID', axis=1, inplace=True)
        self.create_memo('MACD特徴量')

# Ta-Libを利用して算出
class ADX(Feature):
    def create_features(self):
        for id in data.Asset_ID.unique():
        # ADX - Average Directional Movement Index (平均方向性指数)
        # トレンドの存在を確認するための指標
            x = data[data.Asset_ID==id]
            self.data.loc[data.Asset_ID==id, 'adx'] = talib.ADX(x.High, x.Low, x.Close, timeperiod=14)

        self.create_memo('ADX特徴量')


# richmanbtcさんのfeature https://www.kaggle.com/richmanbtc/20211103-gresearch-crypto-v1
# todo: 中身の理解
class Richman_feature(Feature):
    def create_features(self):
        self.data['ln_Close'] = np.log(data['Close'])
        self.data['Weight'] = data.merge(asset_detail[['Asset_ID', 'Weight']], on='Asset_ID', how='left')['Weight']
        self.data['timestamp'] = data['timestamp']
        self.data['Asset_ID'] = data['Asset_ID']

        # shift is faster than diff
        # ログリターンを計算
        self.data['raw_return_causal'] = self.data['ln_Close'] - self.data.groupby('Asset_ID')['ln_Close'].shift(15)
        
        inv_weight_sum = 1.0 / self.data.groupby('timestamp')['Weight'].transform('sum')
        
        self.data['w_raw_return_causal'] = self.data['raw_return_causal'] * self.data['Weight']
        self.data['market_return_causal'] = self.data.groupby('timestamp').w_raw_return_causal.transform('sum') * inv_weight_sum
        
        self.data['raw_market_return_causal'] = self.data['raw_return_causal'] * self.data['market_return_causal']
        self.data['market_return_causal_square'] = self.data['market_return_causal'] ** 2
        self.data['beta_causal'] = (
            self.data.groupby('Asset_ID').raw_market_return_causal.transform(lambda x: x.rolling(3750, 1).mean())
            / self.data.groupby('Asset_ID').market_return_causal_square.transform(lambda x: x.rolling(3750, 1).mean())
        )

        self.data['Close_diff1_rank'] = self.data.groupby('timestamp')['raw_return_causal'].transform('rank')

        self.data.drop(['timestamp', 'Asset_ID'], axis=1, inplace=True)
        self.create_memo('Richmanbtcさんのノートブックの特徴量')

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
