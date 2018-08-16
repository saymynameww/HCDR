import warnings
from multiprocessing import cpu_count

from joblib import Parallel
from bayes_opt import BayesianOptimization
import lightgbm as lgb

import feature as f

warnings.filterwarnings('ignore')


ID_COLUMN = 'SK_ID_CURR'
LABEL_COLUMN = 'TARGET'

n_threads = n_jobs = round(cpu_count() * 2 * 0.85)
n_jobs = cpu_count()
verbose = 1

features = Parallel(n_jobs=cpu_count())(f.get_selected_features_df())

df = None
for i in range(len(features)):
    print('at {} of {}'.format(i, len(features)))
    feature = features[i]
    df = df.join(feature, how='left', on='SK_ID_CURR') if df is not None else feature

print(df.shape)
df.drop(f.columns_not_needed(), axis=1, inplace=True, errors='ignore')

#
# Prepare data
#
train_df = df[df[LABEL_COLUMN].notnull()]
feats = [f for f in train_df.columns if f not in [LABEL_COLUMN, ID_COLUMN, 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
data = lgb.Dataset(data=train_df[feats], label=train_df[LABEL_COLUMN], free_raw_data=False, silent=True)

num_rounds = 10000
num_iter = 1000
init_points = 5
params = {
    'learning_rate': 0.02,
    'verbose_eval': True,
    'nthread': n_threads,
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'device': 'cpu',
}


def lgb_evaluate(max_bin, min_child_samples, learning_rate, num_leaves, min_child_weight, colsample_bytree, max_depth, subsample, reg_lambda, reg_alpha, min_split_gain, random_state):

    params['max_bin'] = int(max(max_bin, 1))
    params['min_child_samples'] = int(max(min_child_samples, 0))
    params['learning_rate'] = max(learning_rate, 0)
    params['num_leaves'] = int(num_leaves)
    params['min_child_weight'] = int(min_child_weight)
    params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['max_depth'] = int(max_depth)
    params['subsample'] = max(min(subsample, 1), 0)
    params['reg_lambda'] = max(reg_lambda, 0)
    params['reg_alpha'] = max(reg_alpha, 0)
    params['min_split_gain'] = max(min_split_gain, 0)

    cv_result = lgb.cv(params,
                       data,
                       nfold=5,
                       num_boost_round=num_rounds,
                       early_stopping_rounds=200,
                       seed=int(random_state),
                       show_stdv=True)

    return max(cv_result['auc-mean'])


lgbBO = BayesianOptimization(lgb_evaluate, {'max_bin': (2**3, 2**11),
                                            'min_child_samples': (10, 50),
                                            'learning_rate': (0.001, 0.5),
                                            'num_leaves': (2, 2 ** 10),
                                            'min_child_weight': (1, 60),
                                            'colsample_bytree': (0.2, 1),
                                            'max_depth': (5, 15),
                                            'subsample': (0.2, 1),
                                            'reg_lambda': (0, 10),
                                            'reg_alpha': (0, 10),
                                            'min_split_gain': (0.01, 0.1),
                                            'random_state': (0, 100000),
                                            })

lgbBO.maximize(init_points=init_points, n_iter=num_iter, acq="poi", xi=0.1)
