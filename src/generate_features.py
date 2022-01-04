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

# https://www.kaggle.com/yamqwe/crypto-prediction-technical-analysis-features
class Technical_analysis_feature(Feature):
    def _rsiFunc(self, prices, n=14):
        deltas = np.diff(prices)
        seed = deltas[:n+1]
        up = seed[seed>=0].sum()/n
        down = -seed[seed<0].sum()/n
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:n] = 100. - 100./(1.+rs)

        for i in range(n, len(prices)):
            delta = deltas[i-1] # cause the diff is 1 shorter

            if delta>0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up*(n-1) + upval)/n
            down = (down*(n-1) + downval)/n

            rs = up/down
            rsi[i] = 100. - 100./(1.+rs)

        return rsi

    def create_features(self):
        df_data = data.copy()

        l_window = [5, 12, 19, 26]
        for i in l_window:
            df_data['moving_average_'+str(i)] = df_data.groupby('Asset_ID').Close.transform(lambda x: x.rolling(window=i).mean())
            df_data['moving_std_'+str(i)] = df_data.groupby('Asset_ID').Close.transform(lambda x: x.rolling(window=i).std())
            
            # 指数移動平均
            df_data['exponential_moving_average_'+str(i)] = df_data.groupby('Asset_ID').Close.transform(lambda x: x.ewm(min_periods=i, span=i).mean())
            
            # Bollinger Band
            df_data['bollinger_band_high_'+str(i)] = df_data['moving_average_'+str(i)] + 2 * df_data['moving_std_'+str(i)]
            df_data['bollinger_band_low_'+str(i)] = df_data['moving_average_'+str(i)] - 2 * df_data['moving_std_'+str(i)]

            # 相対力指数（RSI）
            df_data['RSI_'+str(i)] = df_data.groupby('Asset_ID').Close.transform(lambda x: self._rsiFunc(x.values, i))

            # volumeの移動平均
            df_data['volume_moving_average_'+str(i)] = df_data.groupby('Asset_ID').Volume.transform(lambda x: x.rolling(window=i).mean())

        # 移動平均収束ダイバージェンス（MACD）
        df_data['MACD'] = df_data['exponential_moving_average_12'] - df_data['exponential_moving_average_26']
        df_data['MACD_signal'] = df_data.groupby('Asset_ID').MACD.transform(lambda x: x.rolling(window=9).mean())

        self.data = df_data.drop(data.columns, axis=1)
        self.create_memo('テクニカル分析の際に用いる指標に関する特徴量')

# richmanbtcさんのfeature https://www.kaggle.com/richmanbtc/20211103-gresearch-crypto-v1
# todo: 中身の理解
class Richman_feature(Feature):
    def create_features(self):
        df_data = data.copy()

        df_data['ln_Close'] = np.log(df_data['Close'])
        df_data = df_data.merge(asset_detail[['Asset_ID', 'Weight']], on='Asset_ID', how='left')

        # shift is faster than diff
        # ログリターンを計算
        df_data['raw_return_causal'] = df_data['ln_Close'] - df_data.groupby('Asset_ID')['ln_Close'].shift(15)
        
        inv_weight_sum = 1.0 / df_data.groupby('timestamp')['Weight'].transform('sum')
        
        df_data['w_raw_return_causal'] = df_data['raw_return_causal'] * df_data['Weight']
        df_data['market_return_causal'] = df_data.groupby('timestamp').w_raw_return_causal.transform('sum') * inv_weight_sum
        
        df_data['raw_market_return_causal'] = df_data['raw_return_causal'] * df_data['market_return_causal']
        df_data['market_return_causal_square'] = df_data['market_return_causal'] ** 2
        df_data['beta_causal'] = (
            df_data.groupby('Asset_ID').raw_market_return_causal.transform(lambda x: x.rolling(3750, 1).mean())
            / df_data.groupby('Asset_ID').market_return_causal_square.transform(lambda x: x.rolling(3750, 1).mean())
        )

        df_data['Close_diff1_rank'] = df_data.groupby('timestamp')['raw_return_causal'].transform('rank')

        self.data = df_data.drop(data.columns, axis=1)

        self.data = df_data
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
