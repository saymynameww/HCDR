import warnings
from multiprocessing import cpu_count

import os
from datetime import datetime
import pandas as pd
import numpy as np
from joblib import Parallel
from sklearn.model_selection import train_test_split

import feature as f
import fit_models as fit_models
import utility.expand_grid as expand_grid

warnings.filterwarnings('ignore')


ID_COLUMN = 'SK_ID_CURR'
LABEL_COLUMN = 'TARGET'

n_threads = n_jobs = round(cpu_count() * 2 * 0.75)
n_jobs = cpu_count()
verbose = 1

X, y, X_test, train_test, bureau, bureau_bal, prev, credit_card_bal, pos_cash, installment_payment = f.read_dataset()
feature_mapping = f.get_feature_mapping(train_test, bureau, bureau_bal, prev, credit_card_bal, pos_cash, installment_payment)
features = Parallel(n_jobs=n_jobs, verbose=verbose)(feature_mapping)

for df in features:
    X = X.merge(right=df, how='left', on=ID_COLUMN)
    X_test = X_test.merge(right=df, how='left', on=ID_COLUMN)
    assert X.shape[0] == 307511

print('X.shape', X.shape)
print('X_test.shape', X_test.shape)

#
# Delete customer Id
#

del X['SK_ID_CURR']
test_ID = X_test['SK_ID_CURR']
del X_test['SK_ID_CURR']

#
# Prepare data
#

data = X.copy().reset_index()
data.columns = ['index'] + ['{}_{}'.format(i, c) for i, c in enumerate(data.columns[1:])]
data['label'] = y
data.head()

col_type = dict()
col_type['ID'] = 'index'
col_type['target'] = 'label'
col_type['features'] = [x for x in data.columns
                        if x not in [col_type['target'], col_type['ID']]]

train, test = train_test_split(
    data,
    test_size=0.33,
    random_state=1,
    stratify=data[col_type['target']])

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

#
# Train with LightGBM
#
param_grid = {
    # Boosting parameters
    'learning_rate': [0.1],
    'num_boost_round': [10000],  # this specify the upper bound, we use early_stopping_round to find the optimal value
    'boosting': ['gbdt'],

    # Tree-based parameters
    'num_leaves': [31],
    'min_data_in_leaf': [20],
    'max_depth': [-1],
    'max_bin': [255],  # max number of bins that feature values will be bucketed in

    'bagging_fraction': [0.8],
    'feature_fraction': [0.8],

    # Regulations parameters
    'lambda_l1': [1],
    'lambda_l2': [1],

    # Other parameters
    'is_unbalance': [True],
    'scale_pos_weight': [1.0],
    'device': ['cpu']
}

param_table = expand_grid.expand_grid(param_grid)

# Find the optimal number of trees for this learning rate
param_grid, _ = fit_models.fit_lightgbm(
    param_grid,
    param_table,
    train,
    col_type,
    find_num_boost_round=True,
    cv_iterations=1,
    cv_folds=5,
    nthread=n_threads,
    verbose=verbose
)

#
# Tune num_leaves and min_data_in_leaf
#

param_grid['num_leaves'] = [2 ** x for x in range(3, 10, 2)]
param_grid['min_child_weight'] = range(1, 22, 3)

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_lightgbm(
    param_grid,
    param_table,
    train,
    col_type,
    find_num_boost_round=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=n_threads,
    verbose=verbose
)

current_num_leaves = param_grid['num_leaves'][0]
current_min_child_weight = param_grid['min_child_weight'][0]
param_grid['num_leaves'] = range(
    int(0.8 * current_num_leaves),
    int(1.2 * current_num_leaves),
    int(np.ceil(0.02 * current_num_leaves))
)

param_grid['min_child_weight'] = np.unique(
    range(
        np.max([current_min_child_weight - 1, 1]),
        current_min_child_weight + 4
    )
)

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_lightgbm(
    param_grid,
    param_table,
    train,
    col_type,
    find_num_boost_round=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=n_threads,
    verbose=verbose
)

#
# Tune max_depth
#
current_num_leaves = param_grid['num_leaves'][0]
current_avg_max_depth = int(np.ceil(np.log(current_num_leaves) / np.log(2)))

param_grid['max_depth'] = range(
    np.maximum(2, current_avg_max_depth - 2),
    current_avg_max_depth + 2
)

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_lightgbm(
    param_grid,
    param_table,
    train,
    col_type,
    find_num_boost_round=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=n_threads,
    verbose=verbose
)

#
# Tune max_bin
#
param_grid['max_bin'] = [2 ** x for x in range(3, 11)]

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_lightgbm(
    param_grid,
    param_table,
    train,
    col_type,
    find_num_boost_round=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=n_threads,
    verbose=verbose
)

#
# Tune bagging_fraction and feature_fraction
#
param_grid['bagging_fraction'] = [x / 10.0 for x in range(5, 11)]
param_grid['feature_fraction'] = [x / 10.0 for x in range(5, 11)]

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_lightgbm(
    param_grid,
    param_table,
    train,
    col_type,
    find_num_boost_round=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=n_threads,
    verbose=verbose
)

current_bagging_fraction = param_grid['bagging_fraction'][0]
current_feature_fraction = param_grid['feature_fraction'][0]

param_grid['bagging_fraction'] = [x / 100.0 for x in range(
    int(current_bagging_fraction * 100) - 15,
    np.min([int(current_bagging_fraction * 100) + 15, 105]),
    5
)]

param_grid['feature_fraction'] = [x / 100.0 for x in range(
    int(current_feature_fraction * 100) - 15,
    np.min([int(current_feature_fraction * 100) + 15, 105]),
    5
)]

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_lightgbm(
    param_grid,
    param_table,
    train,
    col_type,
    find_num_boost_round=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=n_threads,
    verbose=verbose
)

#
# Tune lambda_l2
#
param_grid['lambda_l2'] = [0, 1e-5, 0.001, 0.005, 0.01, 0.05, 0.1, 1, 3, 5, 10, 100]

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_lightgbm(
    param_grid,
    param_table,
    train,
    col_type,
    find_num_boost_round=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=n_threads,
    verbose=verbose
)

#
# Tune lambda_l1
#
param_grid['lambda_l1'] = [0, 1e-5, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10, 100]

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_lightgbm(
    param_grid,
    param_table,
    train,
    col_type,
    find_num_boost_round=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=n_threads,
    verbose=verbose
)

#
# Tune Learning Rate
#
param_grid['learning_rate'] = [0.001, 0.01, 0.05, 0.1, 0.5]
param_grid['num_boost_round'] = [1000]

param_table = expand_grid.expand_grid(param_grid)

param_grid, pred = fit_models.fit_lightgbm(
    param_grid,
    param_table,
    train,
    col_type,
    find_num_boost_round=True,
    cv_iterations=5,
    cv_folds=5,
    nthread=n_threads,
    verbose=verbose
)

#
# Final lightGBM parameter and predictions
#
best_param_index = param_table["Score_Weighted"].idxmax()

param_grid_lightgbm = param_grid
pred_lightgbm = pred["Pred_" + str(best_param_index)].rename('pred_lightgbm').reset_index()

print(param_grid_lightgbm)

run_datetime = datetime.now()
submission_file_name = 'lgb_{0:%Y-%m-%d_%H:%M:%S}.csv'.format(run_datetime)
submission = pd.DataFrame({'SK_ID_CURR': test_ID, 'TARGET': pred_lightgbm})
submission.to_csv(submission_file_name, index=False)
