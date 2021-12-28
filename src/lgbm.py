import pandas as pd
import os
import lightgbm as lgbm
import sys
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import csv
import glob
from feature_base import load_datasets
import subprocess

DATA_FOLDER = '../../readonly/final_project_data'
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
    feats = glob.glob(FEATURE_FOLDER+'/**.pkl')
    data = load_datasets(feats)
    data = data.dropna(how="any")

    df_train = data[data['datetime'] < '2021-06-13 00:00:00']
    df_test = data[data['datetime'] >= '2021-06-13 00:00:00']
    if opts.debug:
        df_train = df_train[:1000]

    del_columns = ['datetime', 'Target']
    X = df_train.drop(del_columns, axis=1)
    y = df_train['Target']
    with open(os.path.join(RESULT_FOLDER, 'columns.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(X.columns.to_list())
    
    mlflow.log_artifact(os.path.join(RESULT_FOLDER, 'columns.csv'))

    category_feature = []

    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    lgbm_train = lgbm.Dataset(X_train, y_train)
    lgbm_valid = lgbm.Dataset(X_valid, y_valid)

    print('--- Training ---')

    params = {
        "objective": "regression", 
        "metric": "rmse", 
        "boosting_type": "gbdt",
        'early_stopping_rounds': 50,
        'learning_rate': 0.05,
        'lambda_l1': 1,
        'lambda_l2': 1,
        'max_depth': 7,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        }

    model = lgbm.train(params=params,
                train_set=lgbm_train,
                valid_sets=[lgbm_train, lgbm_valid],
                num_boost_round=5000,
                verbose_eval=100,
                categorical_feature = category_feature,
            )

