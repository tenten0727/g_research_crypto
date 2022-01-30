import pandas as pd
import os
import lightgbm as lgbm
import sys
from sklearn.model_selection import train_test_split
import argparse
import mlflow
import mlflow.lightgbm
import csv
import glob

from feature_base import load_datasets
from utils import weighted_correlation, eval_w_corr

DATA_FOLDER = '../input'
FEATURE_FOLDER = '../features'
RESULT_FOLDER = '../result/LGBM'

parser = argparse.ArgumentParser()
parser.add_argument('--debug',  action='store_true')
parser.add_argument('--save_name', type=str, default='tmp')
opts = parser.parse_args()

if opts.debug:
    opts.save_name = 'debug'

mlflow.lightgbm.autolog()
with mlflow.start_run():
    mlflow.log_params(vars(opts))

    if not os.path.isdir(os.path.join(RESULT_FOLDER, opts.save_name)):
        os.makedirs(os.path.join(RESULT_FOLDER, opts.save_name))

    print('--- Data Preparation ---')
    # feats = glob.glob(FEATURE_FOLDER+'/**.pkl')
    feats = ['Base', 'Arithmetic_operations', 'Shadow_features', 'Richman_feature', 'Volatility_feature']
    feats = [FEATURE_FOLDER+'/'+f+'.pkl' for f in feats]
    data = load_datasets(feats)
    data = data.dropna(subset=['Target'])

    df_train = data[data['datetime'] < '2021-06-13 00:00:00']
    df_test = data[data['datetime'] >= '2021-06-13 00:00:00']
    if opts.debug:
        df_train = df_train[:100000]

    del_columns = ['datetime', 'Target', 'timestamp']
    X = df_train.drop(del_columns, axis=1)
    y = df_train['Target']
    with open(os.path.join(RESULT_FOLDER, 'columns.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(X.columns.to_list())
    
    mlflow.log_artifact(os.path.join(RESULT_FOLDER, 'columns.csv'))

    category_feature = ['Asset_ID']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)
    max_lookback = 30*24*60 + 1
    X_train = X_train[:-max_lookback]
    y_train = y_train[:-max_lookback]
    lgbm_train = lgbm.Dataset(X_train, y_train)
    lgbm_valid = lgbm.Dataset(X_valid, y_valid)
    lgbm_train.add_w = X_train['Weight']
    lgbm_valid.add_w = X_valid['Weight']

    print('--- Training ---')

    params = {
        "objective": "regression", 
        "metric": "rmse", 
        "boosting_type": "gbdt",
        'early_stopping_rounds': 20,
        'learning_rate': 0.05,
        'lambda_l1': 5,
        'lambda_l2': 5,
        'max_depth': 3,
        'num_leaves': 4,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'extra_trees': True,
        }

    mlflow.log_params(params)

    model = lgbm.train(params=params,
                train_set=lgbm_train,
                valid_sets=[lgbm_train, lgbm_valid],
                num_boost_round=1000,
                verbose_eval=10,
                feval=eval_w_corr,
                categorical_feature = category_feature,
            )

    print('--- Test ---')
    X_test = df_test.drop(del_columns, axis=1)
    y_test = df_test[['Target', 'Asset_ID']]
    y_test['predict'] = model.predict(X_test, num_iteration=model.best_iteration)

    asset_details = pd.read_csv(os.path.join(DATA_FOLDER, 'g-research-crypto-forecasting', 'asset_details.csv'))
    weight_map_dict = dict(zip(asset_details['Asset_ID'], asset_details['Weight']))
    y_test['weight'] = y_test['Asset_ID'].map(weight_map_dict)

    metric = weighted_correlation(y_test['predict'], y_test['Target'], y_test['weight'])
    print('weighted_corr:', metric)
    mlflow.log_metric('weighted_corr', metric)

