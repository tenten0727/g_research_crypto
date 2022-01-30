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
from math import sqrt, log

import sys
sys.path.append('../src')
from utils import RunningMean

Feature.dir = '../features'
data, asset_detail = base_data()

def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])

def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']

def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n

def realized_volatility(close, N=240):
    rt = list(log(C_t / C_t_1) for C_t, C_t_1 in zip(close[1:], close[:-1]))
    rt_mean = sum(rt) / len(rt)
    return sqrt(sum((r_i - rt_mean) ** 2 for r_i in rt) * N / (len(rt) - 1))

class Base(Feature):
    def create_features(self):
        self.data = data.copy()
        self.data['Weight'] = data.merge(asset_detail[['Asset_ID', 'Weight']], on='Asset_ID', how='left')['Weight']

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

        self.data['hlco_ration'] = (data["Open"] - data["Close"]) / (data["High"] - data["Low"])

        # self.data['mean_price'] = data[['High', 'Low', 'Open', 'Close']].mean(axis=1)
        # self.data['median_price'] = data[['High', 'Low', 'Open', 'Close']].median(axis=1)

        self.create_memo('四則演算で計算した特徴量')

# https://www.kaggle.com/yamqwe/crypto-prediction-technical-analysis-features
class Window_feature(Feature):
    def create_features(self):
        self.data['Asset_ID'] = data['Asset_ID']
        self.data['timestamp'] = data['timestamp']

        asset_group_close = data.groupby('Asset_ID').Close
        l_window = [5, 15, 60]
        for i in l_window:
            self.data['moving_average_'+str(i)] = asset_group_close.transform(lambda x: moving_average(x.values, i))
            self.data['moving_std_'+str(i)] = asset_group_close.transform(lambda x: x.rolling(window=i, min_periods=1).std())
            
            # 指数移動平均
            # self.data['exponential_moving_average_'+str(i)] = asset_group_close.transform(lambda x: x.ewm(min_periods=i, span=i).mean())
            
            # volumeの移動平均
            self.data['volume_moving_average_'+str(i)] = data.groupby('Asset_ID').Volume.transform(lambda x: moving_average(x.values, i))
            
            # Bollinger Band
            # self.data['bollinger_band_high_'+str(i)] = self.data['moving_average_'+str(i)] + 2 * self.data['moving_std_'+str(i)]
            # self.data['bollinger_band_low_'+str(i)] = self.data['moving_average_'+str(i)] - 2 * self.data['moving_std_'+str(i)]

            # 相対力指数（RSI）...相場の過熱感を一定期間の終値から計算するオシレータ系指標
            self.data['RSI_'+str(i)] = asset_group_close.transform(lambda x: talib.RSI(x.values, i))

            # ２重指数移動平均(DEMA: Double Exponential Moving Average)
            # self.data['double_ema_'+str(i)] = asset_group_close.transform(lambda x: talib.DEMA(x, timeperiod=i))

            # Kaufmanの適応型移動平均(KAMA: Kaufman Adaptive Moving Average)
            # self.data['kama_'+str(i)] = asset_group_close.transform(lambda x: talib.KAMA(x, timeperiod=i))

        self.data['close_div_ma_5'] = data['Close'] / self.data['moving_average_5']
        self.data['close_div_ma_15'] = data['Close'] / self.data['moving_average_15']
        self.data['close_div_ma_60'] = data['Close'] / self.data['moving_average_60']

        self.data['volume_div_ma_5'] = data['Volume'] / self.data['volume_moving_average_5']
        self.data['volume_div_ma_15'] = data['Volume'] / self.data['volume_moving_average_15']
        self.data['volume_div_ma_60'] = data['Volume'] / self.data['volume_moving_average_60']

        self.data['close_div_ma_15_rank'] = self.data.groupby('timestamp').close_div_ma_15.transform('rank')
        self.data['volume_div_ma_15_rank'] = self.data.groupby('timestamp').volume_div_ma_15.transform('rank')

        self.data['RSI_5_rank'] = self.data.groupby('timestamp').RSI_5.transform('rank')
        self.data['RSI_15_rank'] = self.data.groupby('timestamp').RSI_15.transform('rank')
        self.data['RSI_60_rank'] = self.data.groupby('timestamp').RSI_60.transform('rank')

        ma = [col for col in self.data.columns if 'moving_average_' in col]
        vol = [col for col in self.data.columns if 'volume_moving_average_' in col]
        self.data.drop(['Asset_ID', 'timestamp']+ma+vol, axis=1, inplace=True)
        self.create_memo('テクニカル分析の際に用いる指標に関する特徴量')

# Ta-Libを利用して算出
# class Trend_Line(Feature):
#     def create_features(self):
#         asset_group_close = data.groupby('Asset_ID').Close

#         # トレンドライン(Hilbert Transform - Instantaneous Trendline)
#         self.data['ht_trendline'] = asset_group_close.transform(lambda x: talib.HT_TRENDLINE(x.values))

#         self.create_memo('trend_line特徴量')

class Per_Asset_feature(Feature):
    def create_features(self):
        self.data['Asset_ID'] = data['Asset_ID']

        for i in data.Asset_ID.unique():
            self.data.loc[self.data.Asset_ID==i, 'MACD'], self.data.loc[self.data.Asset_ID==i, 'MACD_signal'], self.data.loc[self.data.Asset_ID==i, 'MACD_hist'] = talib.MACD(data[data.Asset_ID==i].Close.values, fastperiod=12, slowperiod=26, signalperiod=9)

            x = data[data.Asset_ID==i]
            self.data.loc[self.data.Asset_ID==i, 'adx'] = talib.ADX(x.High.values, x.Low.values, x.Close.values, timeperiod=14)

        self.data.drop('Asset_ID', axis=1, inplace=True)
        self.create_memo('Asset_IDごとの特徴量')


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
        self.data['log_return_5'] = self.data['ln_Close'] - self.data.groupby('Asset_ID')['ln_Close'].shift(5)
        self.data['log_return_15'] = self.data['ln_Close'] - self.data.groupby('Asset_ID')['ln_Close'].shift(15)
        self.data['log_return_60'] = self.data['ln_Close'] - self.data.groupby('Asset_ID')['ln_Close'].shift(60)
        
        inv_weight_sum = 1.0 / self.data.groupby('timestamp')['Weight'].transform('sum')
        
        self.data['w_log_return_15'] = self.data['log_return_15'] * self.data['Weight']
        self.data['market_return_causal'] = self.data.groupby('timestamp').w_log_return_15.transform('sum') * inv_weight_sum
        
        self.data['raw_market_return_causal'] = self.data['log_return_15'] * self.data['market_return_causal']
        self.data['market_return_causal_square'] = self.data['market_return_causal'] ** 2
        self.data['beta_causal'] = (
            self.data.groupby('Asset_ID').raw_market_return_causal.transform(lambda x: moving_average(x.fillna(0).values, 60))
            / self.data.groupby('Asset_ID').market_return_causal_square.transform(lambda x: moving_average(x.fillna(0).values, 60))
        )

        self.data['Close_diff1_rank'] = self.data.groupby('timestamp')['log_return_15'].transform('rank')

        self.data.drop(['timestamp', 'Asset_ID', 'Weight', 'ln_Close'], axis=1, inplace=True)
        self.create_memo('Richmanbtcさんのノートブックの特徴量')

class Volatility_feature(Feature):
    def create_features(self):
        self.data['Asset_ID'] = data['Asset_ID']
        self.data['ln_Close'] = np.log(data['Close'])
        self.data['timestamp'] = data['timestamp']

        self.data['log_return_1'] = self.data.ln_Close - self.data.groupby('Asset_ID')['ln_Close'].shift(1)
        for i in [5, 15, 60]:
            self.data['realized_volatility_'+str(i)] = self.data.groupby('Asset_ID').log_return_1.transform(lambda x: x.rolling(i).std(ddof=0))
            self.data['RV_'+str(i)+'_rank'] = self.data.groupby('timestamp')['realized_volatility_'+str(i)].transform('rank')

        self.data.drop(['Asset_ID', 'ln_Close', 'timestamp'], axis=1, inplace=True)
        self.create_memo('Volatilityに関する特徴量')

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
