import pickle
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
import argparse
import mlflow
import csv
import glob
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler


from feature_base import load_datasets
from utils import weighted_correlation, eval_w_corr

DATA_FOLDER = '../input'
FEATURE_FOLDER = '../features'
RESULT_FOLDER = '../result/Ridge'

parser = argparse.ArgumentParser()
parser.add_argument('--debug',  action='store_true')
parser.add_argument('--save_name', type=str, default='tmp')
opts = parser.parse_args()

if opts.debug:
    opts.save_name = 'debug'

with mlflow.start_run(experiment_id=2):
    mlflow.log_params(vars(opts))

    if not os.path.isdir(os.path.join(RESULT_FOLDER, opts.save_name)):
        os.makedirs(os.path.join(RESULT_FOLDER, opts.save_name))

    print('--- Data Preparation ---')
    feats = glob.glob(FEATURE_FOLDER+'/**.pkl')
    data = load_datasets(feats)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(how="any")

    df_train = data[data['datetime'] < '2021-06-13 00:00:00']
    df_test = data[data['datetime'] >= '2021-06-13 00:00:00']
    if opts.debug:
        df_train = df_train[:100000]

    del_columns = ['datetime', 'Target', 'timestamp', 'Asset_ID']
    X = df_train.drop(del_columns, axis=1)
    y = df_train['Target']
    with open(os.path.join(RESULT_FOLDER, 'columns.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(X.columns.to_list())
    
    mlflow.log_artifact(os.path.join(RESULT_FOLDER, 'columns.csv'))

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)
    max_lookback = 30*24*60 + 1
    X_train = X_train[:-max_lookback]
    y_train = y_train[:-max_lookback]

    print('--- Training ---')

    model = Ridge()

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_valid)

    val_score = weighted_correlation(pred, y_valid, X_valid['Weight'])
    print('val_score: ', val_score)
    mlflow.log_metric('val_score', val_score)

    pickle.dump(model, open(os.path.join(RESULT_FOLDER, opts.save_name, 'model.pkl'), 'wb'))

    print('--- Test ---')
    X_test = df_test.drop(del_columns, axis=1)
    y_test = df_test[['Target', 'Asset_ID', 'Weight']]
    y_test['predict'] = model.predict(X_test)

    # asset_details = pd.read_csv(os.path.join(DATA_FOLDER, 'g-research-crypto-forecasting', 'asset_details.csv'))
    # weight_map_dict = dict(zip(asset_details['Asset_ID'], asset_details['Weight']))
    # y_test['Weight'] = y_test['Asset_ID'].map(weight_map_dict)

    metric = weighted_correlation(y_test['predict'], y_test['Target'], y_test['Weight'])
    print('weighted_corr:', metric)
    mlflow.log_metric('weighted_corr', metric)

