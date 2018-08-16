import warnings
from multiprocessing import cpu_count

import numpy as np
from joblib import Parallel
from scipy.stats import norm
from sklearn.model_selection import train_test_split

import feature as f
import fit_models as fit_models
import utility.expand_grid as expand_grid
import utility.random_grid as random_grid

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
# Train with XGBoost
#
param_grid = {
    # Boosting parameters
    'learning_rate': [0.1],
    'n_estimators': [2000],  # this specify the upper bound, we use early stop to find the optimal value

    # Tree-based parameters
    'max_depth': [6],
    'min_child_weight': [1],
    'gamma': [0],
    'subsample': [0.8],
    'colsample_bytree': [0.8],

    # Regulations parameters
    'reg_lambda': [1],
    'reg_alpha': [1],

    # Other parameters
    'scale_pos_weight': [1]
}

param_table = expand_grid.expand_grid(param_grid)

# Find the optimal number of trees for this learning rate
param_grid, _ = fit_models.fit_xgboost(
    param_grid,
    param_table,
    train,
    col_type,
    find_n_estimator=True,
    cv_iterations=1,
    cv_folds=5,
    nthread=n_threads,
    verbose=verbose
)

param_grid['max_depth'] = range(3, 20, 2)
param_grid['min_child_weight'] = range(1, 6, 2)

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_xgboost(
    param_grid,
    param_table,
    train,
    col_type,
    find_n_estimator=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=n_threads,
    verbose=verbose
)

# Fine search
current_max_depth = param_grid['max_depth'][0]
current_min_child_weight = param_grid['min_child_weight'][0]
param_grid['max_depth'] = np.unique([
    np.max([current_max_depth - 1, 1]),
    current_max_depth,
    current_max_depth + 1
])
param_grid['min_child_weight'] = np.unique([
    np.max([current_min_child_weight - 1, 1]),
    current_min_child_weight,
    current_min_child_weight + 1
])

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_xgboost(
    param_grid,
    param_table,
    train,
    col_type,
    find_n_estimator=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=n_threads,
    verbose=verbose
)

param_grid['gamma'] = [x / 10.0 for x in range(0, 15)]

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_xgboost(
    param_grid,
    param_table,
    train,
    col_type,
    find_n_estimator=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=n_threads,
    verbose=verbose
)

# Coarse search
param_grid['subsample'] = [x / 10.0 for x in range(5, 11)]
param_grid['colsample_bytree'] = [x / 10.0 for x in range(5, 11)]

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_xgboost(
    param_grid,
    param_table,
    train,
    col_type,
    find_n_estimator=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=n_threads,
    verbose=verbose
)

# Fine search
current_subsample = param_grid['subsample'][0]
current_colsample_bytree = param_grid['colsample_bytree'][0]

param_grid['subsample'] = [x / 100.0 for x in range(
    int(current_subsample * 100) - 15,
    np.min([int(current_subsample * 100) + 15, 105]),
    5
)]

param_grid['colsample_bytree'] = [x / 100.0 for x in range(
    int(current_colsample_bytree * 100) - 15,
    np.min([int(current_colsample_bytree * 100) + 15, 105]),
    5
)]

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_xgboost(
    param_grid,
    param_table,
    train,
    col_type,
    find_n_estimator=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=n_threads,
    verbose=verbose
)

param_grid['reg_lambda'] = [0, 1e-5, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10, 100]

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_xgboost(
    param_grid,
    param_table,
    train,
    col_type,
    find_n_estimator=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=n_threads,
    verbose=verbose
)

param_grid['reg_alpha'] = [0, 1e-5, 0.001, 0.005, 0.01, 0.05, 0.1, 1, 3, 5, 10, 100]

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_xgboost(
    param_grid,
    param_table,
    train,
    col_type,
    find_n_estimator=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=n_threads,
    verbose=verbose
)

print(param_grid)

param_grid['max_depth'] = [8, 9, 10]
param_grid['min_child_weight'] = [1, 2]
param_grid['gamma'] = [1.1, 1.2, 1.3]
param_grid['subsample'] = norm(loc=param_grid['subsample'][0], scale=0.02)
param_grid['colsample_bytree'] = norm(loc=param_grid['colsample_bytree'][0], scale=0.02)
param_grid['reg_lambda'] = norm(loc=param_grid['reg_lambda'][0], scale=0.02)
param_grid['reg_alpha'] = norm(loc=param_grid['reg_alpha'][0], scale=0.0001)

param_table = random_grid.random_grid(
    param_grid,
    random_iter=100,
    random_state=1
)

param_table['reg_lambda'][param_table['reg_lambda'] < 0] = 0
param_table['reg_alpha'][param_table['reg_alpha'] < 0] = 0

param_grid, _ = fit_models.fit_xgboost(
    param_grid,
    param_table,
    train,
    col_type,
    find_n_estimator=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=n_threads,
    verbose=verbose
)


param_grid['learning_rate'] = [0.001, 0.01, 0.05, 0.1, 0.5]
param_grid['n_estimators'] = [1000]

param_table = expand_grid.expand_grid(param_grid)

param_grid, pred = fit_models.fit_xgboost(
    param_grid,
    param_table,
    train,
    col_type,
    find_n_estimator=True,
    cv_iterations=5,
    cv_folds=5,
    nthread=n_threads,
    verbose=verbose
)

best_param_index = param_table["Score_Weighted"].idxmax()

param_grid_xgboost = param_grid
pred_xgboost = pred["Pred_" + str(best_param_index)].rename('pred_xgboost').reset_index()

print(param_grid_xgboost)
