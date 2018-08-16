# Forked from excellent kernel : https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features
# From Kaggler : https://www.kaggle.com/jsaguiar
# Just added a few features so I thought I had to make release it as well...

import gc
import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from joblib import Parallel
from multiprocessing import cpu_count
from datetime import datetime

import feature as f

LABEL_COLUMN = 'TARGET'

DATA_DIR = '{}/data'.format(os.getcwd())
SUBMISSION_DIR = '{}/submission'.format(os.getcwd())


def kfold_lightgbm(df, num_folds, stratified=False, debug=False):
    train_df = df[df[LABEL_COLUMN].notnull()]
    test_df = df[df[LABEL_COLUMN].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()

    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)

    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in [LABEL_COLUMN, 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df[LABEL_COLUMN])):
        dtrain = lgb.Dataset(data=train_df[feats].iloc[train_idx],
                             label=train_df[LABEL_COLUMN].iloc[train_idx],
                             free_raw_data=False, silent=True)
        dvalid = lgb.Dataset(data=train_df[feats].iloc[valid_idx],
                             label=train_df[LABEL_COLUMN].iloc[valid_idx],
                             free_raw_data=False, silent=True)

        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'nthread': 4,
            'learning_rate': 0.02,  # 02,
            'num_leaves': 20,
            'colsample_bytree': 0.9497036,
            'subsample': 0.8715623,
            'subsample_freq': 1,
            'max_depth': 8,
            'reg_alpha': 0.041545473,
            'reg_lambda': 0.0735294,
            'min_split_gain': 0.0222415,
            'min_child_weight': 60,  # 39.3259775,
            'seed': 0,
            'verbose': -1,
            'metric': 'auc',
        }

        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=10000,
            valid_sets=[dtrain, dvalid],
            early_stopping_rounds=200,
            verbose_eval=False
        )

        oof_preds[valid_idx] = clf.predict(dvalid.data)
        sub_preds += clf.predict(test_df[feats]) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0, sort=True)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(dvalid.label, oof_preds[valid_idx])))
        del clf, dtrain, dvalid
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df[LABEL_COLUMN], oof_preds))

    if not debug:
        sub_df = test_df[['SK_ID_CURR']].copy()
        sub_df[LABEL_COLUMN] = sub_preds
        sub_df[['SK_ID_CURR', LABEL_COLUMN]].to_csv(submission_file_name, index=False)
    display_importances(feature_importance_df)
    return feature_importance_df


def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout

    feature_importance_file_name = os.path.join(SUBMISSION_DIR, 'ogrellier_feat_{0:%Y-%m-%d_%H:%M:%S}.csv'.format(run_datetime))
    plt.savefig(feature_importance_file_name)


def main(debug=False):
    features = Parallel(n_jobs=cpu_count())(f.get_selected_features_df())

    df = None
    for i in range(len(features)):
        print('at {} of {}'.format(i, len(features)))
        feature = features[i]
        df = df.join(feature, how='left', on='SK_ID_CURR') if df is not None else feature

    print(df.shape)
    df.drop(f.columns_not_needed(), axis=1, inplace=True, errors='ignore')
    feat_importance = kfold_lightgbm(df, num_folds=5, stratified=False, debug=debug)
    display_importances(feat_importance)


if __name__ == "__main__":
    run_datetime = datetime.now()
    submission_file_name = os.path.join(SUBMISSION_DIR, 'ogrellier_{0:%Y-%m-%d_%H:%M:%S}.csv'.format(run_datetime))
    main()
