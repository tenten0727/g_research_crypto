import pandas as pd
import os
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import sys
from sklearn.model_selection import train_test_split
import argparse
import mlflow
import mlflow.lightgbm
import csv
import glob
import numpy as np
import pickle

from feature_base import load_datasets
from utils import weighted_correlation, eval_w_corr, PurgedGroupTimeSeriesSplit

DATA_FOLDER = '../input'
FEATURE_FOLDER = '../features'
RESULT_FOLDER = '../result/LGBM_kfold'

parser = argparse.ArgumentParser()
parser.add_argument('--debug',  action='store_true')
parser.add_argument('--save_name', type=str, default='tmp')
opts = parser.parse_args()

if opts.debug:
    opts.save_name = 'debug'

mlflow.lightgbm.autolog()
with mlflow.start_run(experiment_id=4):
    mlflow.log_params(vars(opts))

    if not os.path.isdir(os.path.join(RESULT_FOLDER, opts.save_name)):
        os.makedirs(os.path.join(RESULT_FOLDER, opts.save_name))

    print('--- Data Preparation ---')
    # feats = glob.glob(FEATURE_FOLDER+'/**.pkl')
    feats = ['Base', 'Arithmetic_operations', 'Shadow_features', 'Richman_feature', 'Window_feature', 'Volatility_feature']
    feats = [FEATURE_FOLDER+'/'+f+'.pkl' for f in feats]
    data = load_datasets(feats)
    data = data.dropna(subset=['Target'])

    df_train = data[data['datetime'] < '2021-06-13 00:00:00']
    df_test = data[data['datetime'] >= '2021-06-13 00:00:00']
    if opts.debug:
        df_train = df_train[:1000000]

    del_columns = ['datetime', 'Target', 'timestamp']
    X = df_train.drop(del_columns, axis=1)
    y = df_train['Target']
    with open(os.path.join(RESULT_FOLDER, 'columns.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(X.columns.to_list())
    
    mlflow.log_artifact(os.path.join(RESULT_FOLDER, 'columns.csv'))

    category_feature = ['Asset_ID']

    n_fold = 5
    cv = PurgedGroupTimeSeriesSplit(n_splits = n_fold, group_gap = 50)
    groups = pd.factorize(df_train['datetime'].dt.day.astype(str) + '_' + df_train['datetime'].dt.month.astype(str) + '_' + df_train['datetime'].dt.year.astype(str))[0]
    asset_details = pd.read_csv(os.path.join(DATA_FOLDER, 'g-research-crypto-forecasting', 'asset_details.csv'))
    weight_map_dict = dict(zip(asset_details['Asset_ID'], asset_details['Weight']))

    models = []
    oof_preds = np.zeros(len(X))

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)
    max_lookback = 30*24*60 + 1
    X_train = X_train[:-max_lookback]
    y_train = y_train[:-max_lookback]

    for i, (trn, val) in enumerate(cv.split(X, y, groups)):
        with mlflow.start_run(experiment_id=4, nested=True):
            print('fold: ', i)
            mlflow.log_param('fold', i)

            model = Ridge(alpha=100)

            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            models.append(model)
            pred = model.predict(X_valid, num_iteration=model.best_iteration)
            oof_preds[val] = pred
            cv_score = weighted_correlation(pred, y_valid, X_valid.Asset_ID.map(weight_map_dict))
            print(f'~~~~~~~~ FOLD {i} wcorr: {cv_score} ~~~~~~~~')
            model.save_model(os.path.join(RESULT_FOLDER, opts.save_name, f'model{i}.lgb'), num_iteration=model.best_iteration)
            mlflow.log_metric('fold_score', cv_score)
    pickle.dump(oof_preds, os.path.join(RESULT_FOLDER, opts.save_name, 'oof_preds.pkl'))
    

    print('--- Test ---')
    X_test = df_test.drop(del_columns, axis=1)
    y_test = df_test[['Target', 'Asset_ID']]
    for model in models:
        y_test['predict'] = model.predict(X_test, num_iteration=model.best_iteration) / n_fold

    y_test['weight'] = y_test['Asset_ID'].map(weight_map_dict)

    metric = weighted_correlation(y_test['predict'], y_test['Target'], y_test['weight'])
    print('weighted_corr:', metric)
    mlflow.log_metric('weighted_corr', metric)

