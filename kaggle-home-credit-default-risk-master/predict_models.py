import pandas as pd
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression


def predict_xgboost(param_grid, train, test, col_type, nthread=3, seed=1):
    target = col_type['target']
    features = col_type['features']
    ID = col_type['ID']

    params = dict()
    for key, value in param_grid.items():
        params[key] = value[0]

    params['objective'] = 'binary:logistic'
    params['nthread'] = nthread
    params['random_state'] = seed
    params['seed'] = seed
    params['silent'] = True

    xgb_model = XGBClassifier()
    xgb_model.set_params(**params)

    X_train = train[features].values
    y_train = train[target].values
    X_test = test[features].values

    # Fit the algorithm on train data
    xgb_model.fit(X_train, y_train, eval_metric='auc')

    # Predict on test data
    pred = xgb_model.predict_proba(X_test)[:, 1]

    pred = pd.concat([
        test.loc[:, [ID]],
        pd.Series(pred, name='pred_xgboost')
    ], axis=1)

    return pred


def predict_lightgbm(param_grid, train, test, col_type, nthread=3, seed=1):
    target = col_type['target']
    features = col_type['features']
    ID = col_type['ID']

    params = dict()
    for key, value in param_grid.items():
        params[key] = value[0]

    params['objective'] = 'binary'
    params['num_threads'] = nthread
    params['data_random_seed'] = seed
    params['feature_fraction_seed'] = seed
    params['bagging_seed'] = seed
    params['verbosity'] = 0

    lgb_model = lgb.LGBMClassifier()
    lgb_model.set_params(**params)

    X_train = train[features].values
    y_train = train[target].values
    X_test = test[features].values

    # Fit the algorithm on train data
    lgb_model.fit(X_train, y_train, eval_metric='auc')

    # Predict on test data
    pred = lgb_model.predict_proba(X_test)[:, 1]

    pred = pd.concat([
        test.loc[:, [ID]],
        pd.Series(pred, name='pred_lightgbm')
    ], axis=1)

    return pred


def predict_logistic_regression(param_grid, train, test, col_type, nthread=3, seed=1):
    target = col_type['target']
    features = col_type['features']
    ID = col_type['ID']

    params = dict()
    for key, value in param_grid.items():
        params[key] = value[0]

    params['n_jobs'] = nthread
    params['random_state'] = seed
    params['verbose'] = 0

    lr_model = LogisticRegression()
    lr_model.set_params(**params)

    X_train = train[features].values
    y_train = train[target].values
    X_test = test[features].values

    # Fit the algorithm on train data
    lr_model.fit(X_train, y_train)

    # Predict on test data
    pred = lr_model.predict_proba(X_test)[:, 1]

    pred = pd.concat([
        test.loc[:, [ID]],
        pd.Series(pred, name='pred_logistic_regression')
    ], axis=1)

    return pred