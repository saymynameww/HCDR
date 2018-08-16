import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from time import gmtime, strftime

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression


def fit_xgboost(param_grid, param_table, train, col_type, find_n_estimator=False,
                cv_iterations=5, cv_folds=5, nthread=3, seed=1, verbose=0):

    target = col_type['target']
    features = col_type['features']
    ID = col_type['ID']

    start_time = strftime("%Y-%m-%d %H-%M", gmtime())
    pred_return = {}
    for params in param_table.itertuples(index=True, name='NamedTuple'):
        params = params._asdict()
        index = params['Index']
        params.pop('Index')  # remove "Index" from params

        params['objective'] = 'binary:logistic'
        params['nthread'] = nthread
        params['random_state'] = seed
        params['seed'] = seed
        params['silent'] = True

        xgb_model = XGBClassifier()
        xgb_model.set_params(**params)

        if find_n_estimator:
            xgb_train = xgb.DMatrix(train[features], label=train[target])
            cv_result = xgb.cv(
                xgb_model.get_xgb_params(),
                xgb_train,
                num_boost_round=int(params['n_estimators']),
                nfold=cv_folds,
                metrics='auc',
                early_stopping_rounds=50,
                seed=seed)

            best_n_estimator = cv_result.shape[0]
            param_table.at[index, 'n_estimators'] = best_n_estimator
            xgb_model.set_params(n_estimators=best_n_estimator)

        scores = []
        pred_all = []
        for cv_index in range(cv_iterations):
            pred = train.loc[:, [ID]]  # get only the ID column
            # k-fold cross validation
            skf = StratifiedKFold(n_splits=cv_folds, random_state=cv_index, shuffle=True)

            for train_index, dev_index in skf.split(train[features].values, train[target].values):
                X_train = train[features].iloc[train_index].values
                y_train = train[target].iloc[train_index].values

                X_dev = train[features].iloc[dev_index].values
                y_dev = train[target].iloc[dev_index].values

                # Fit the algorithm on train folds
                xgb_model.fit(X_train, y_train, eval_metric='auc')

                # Predict on dev fold
                pred_dev = xgb_model.predict_proba(X_dev)[:, 1]
                pred.at[dev_index, 'Pred'] = pred_dev

                # Compute the score
                score = metrics.roc_auc_score(y_dev, pred_dev)
                scores.append(score)

            if len(pred_all) == 0:
                pred_all = pred
            else:
                pred_all = pd.concat([pred_all, pred], axis=0)

        pred_mean = pred_all.groupby(ID)['Pred'].mean()  # avg predict_proba for each ID
        score = metrics.roc_auc_score(train.sort_values(ID)[target].values,
                                      pred_mean)  # use avg pred to compute auc score
        pred_return['Pred_' + str(index)] = pred_mean  # store the pred result for use in stacking

        param_table.at[index, 'Score'] = score
        param_table.at[index, 'Score_Std'] = np.std(scores)

        if verbose == 1:
            print('{} : {}'.format(index, param_table.iloc[index, :]))

    param_table["Score_Weighted"] = param_table["Score"] - 0.1 * param_table["Score_Std"]

    # update_param_grid
    best_param_index = param_table["Score_Weighted"].idxmax()
    print("Param_grid size: {}".format(param_table.shape[0]))
    print("Current Score: {},  Score_Std: {}".format(param_table.loc[best_param_index, "Score"],
                                                     param_table.loc[best_param_index, "Score_Std"]))
    print("--------------------------")
    for param in param_grid:
        best_param = param_table.loc[best_param_index, param]
        if isinstance(param_grid[param], list):
            if len(param_grid[param]) > 1 or (len(param_grid[param]) == 1 and param_grid[param][0] != best_param):
                print("{}: tuned to {}".format(param, best_param))
        else:
            print("{}: tuned to {}".format(param, best_param))
        param_grid[param] = [best_param]

    return param_grid, pred_return


def fit_lightgbm(param_grid, param_table, train, col_type, find_num_boost_round=False,
                 cv_iterations=5, cv_folds=5, nthread=3, seed=1, verbose=0):

    target = col_type['target']
    features = col_type['features']
    ID = col_type['ID']

    start_time = strftime("%Y-%m-%d %H-%M", gmtime())
    pred_return = {}
    for params in param_table.itertuples(index=True, name='NamedTuple'):
        params = params._asdict()
        index = params['Index']
        params.pop('Index')  # remove "Index" from params

        params['objective'] = 'binary'
        params['num_threads'] = nthread
        params['data_random_seed'] = seed
        params['feature_fraction_seed'] = seed
        params['bagging_seed'] = seed
        params['verbosity'] = verbose

        lgb_model = lgb.LGBMClassifier()
        lgb_model.set_params(**params)

        if find_num_boost_round:
            lgb_train = lgb.Dataset(train[features], label=train[target])
            cv_result = lgb.cv(
                lgb_model.get_params(),
                lgb_train,
                num_boost_round=int(params['num_boost_round']),
                nfold=cv_folds,
                metrics='auc',
                early_stopping_rounds=50,
                seed=seed,
                verbose_eval=False
            )

            best_num_boost_round = len(cv_result['auc-mean'])

            param_table.at[index, 'num_boost_round'] = best_num_boost_round
            lgb_model.set_params(num_boost_round=best_num_boost_round)

        scores = []
        pred_all = []
        for cv_index in range(cv_iterations):
            pred = train.loc[:, [ID]]  # get only the ID column
            # k-fold cross validation
            skf = StratifiedKFold(n_splits=cv_folds, random_state=cv_index, shuffle=True)

            for train_index, dev_index in skf.split(train[features].values, train[target].values):
                X_train = train[features].iloc[train_index].values
                y_train = train[target].iloc[train_index].values

                X_dev = train[features].iloc[dev_index].values
                y_dev = train[target].iloc[dev_index].values

                # Fit the algorithm on train folds
                lgb_model.fit(X_train, y_train, eval_metric='auc')

                # Predict on dev fold
                pred_dev = lgb_model.predict_proba(X_dev)[:, 1]
                pred.at[dev_index, 'Pred'] = pred_dev

                # Compute the score
                score = metrics.roc_auc_score(y_dev, pred_dev)
                scores.append(score)

            if len(pred_all) == 0:
                pred_all = pred
            else:
                pred_all = pd.concat([pred_all, pred], axis=0)

        pred_mean = pred_all.groupby(ID)['Pred'].mean()  # avg predict_proba for each ID
        score = metrics.roc_auc_score(train.sort_values(ID)[target].values,
                                      pred_mean)  # use avg pred to compute auc score
        pred_return['Pred_' + str(index)] = pred_mean  # store the pred result for use in stacking

        param_table.at[index, 'Score'] = score
        param_table.at[index, 'Score_Std'] = np.std(scores)

        if verbose == 1:
            print('{} : {}'.format(index, param_table.iloc[index, :]))

    param_table["Score_Weighted"] = param_table["Score"] - 0.1 * param_table["Score_Std"]

    # update_param_grid
    best_param_index = param_table["Score_Weighted"].idxmax()
    print("Param_grid size: {}".format(param_table.shape[0]))
    print("Current Score: {},  Score_Std: {}".format(param_table.loc[best_param_index, "Score"],
                                                     param_table.loc[best_param_index, "Score_Std"]))
    print("--------------------------")
    for param in param_grid:
        best_param = param_table.loc[best_param_index, param]
        if isinstance(param_grid[param], list):
            if len(param_grid[param]) > 1 or (len(param_grid[param]) == 1 and param_grid[param][0] != best_param):
                print("{}: tuned to {}".format(param, best_param))
        else:
            print("{}: tuned to {}".format(param, best_param))
        param_grid[param] = [best_param]

    return param_grid, pred_return


def fit_logistic_regression(param_grid, param_table, train, col_type,
                            cv_iterations=5, cv_folds=5, nthread=3, seed=1, verbose=0):

    target = col_type['target']
    features = col_type['features']
    ID = col_type['ID']

    start_time = strftime("%Y-%m-%d %H-%M", gmtime())
    pred_return = {}
    for params in param_table.itertuples(index=True, name='NamedTuple'):
        params = params._asdict()
        index = params['Index']
        params.pop('Index')  # remove "Index" from params

        params['n_jobs'] = nthread
        params['random_state'] = seed
        params['verbose'] = verbose

        lr_model = LogisticRegression()
        lr_model.set_params(**params)

        scores = []
        pred_all = []
        for cv_index in range(cv_iterations):
            pred = train.loc[:, [ID]]  # get only the ID column
            # k-fold cross validation
            skf = StratifiedKFold(n_splits=cv_folds, random_state=cv_index, shuffle=True)

            for train_index, dev_index in skf.split(train[features].values, train[target].values):
                X_train = train[features].iloc[train_index].values
                y_train = train[target].iloc[train_index].values

                X_dev = train[features].iloc[dev_index].values
                y_dev = train[target].iloc[dev_index].values

                # Fit the algorithm on train folds
                lr_model.fit(X_train, y_train)

                # Predict on dev fold
                pred_dev = lr_model.predict_proba(X_dev)[:, 1]
                pred.at[dev_index, 'Pred'] = pred_dev

                # Compute the score
                score = metrics.roc_auc_score(y_dev, pred_dev)
                scores.append(score)

            if len(pred_all) == 0:
                pred_all = pred
            else:
                pred_all = pd.concat([pred_all, pred], axis=0)

        pred_mean = pred_all.groupby(ID)['Pred'].mean()  # avg predict_proba for each ID
        score = metrics.roc_auc_score(train.sort_values(ID)[target].values,
                                      pred_mean)  # use avg pred to compute auc score
        pred_return['Pred_' + str(index)] = pred_mean  # store the pred result for use in stacking

        param_table.at[index, 'Score'] = score
        param_table.at[index, 'Score_Std'] = np.std(scores)

        if verbose == 1:
            print('{} : {}'.format(index, param_table.iloc[index, :]))

    param_table["Score_Weighted"] = param_table["Score"] - 0.1 * param_table["Score_Std"]

    # update_param_grid
    best_param_index = param_table["Score_Weighted"].idxmax()
    print("Param_grid size: {}".format(param_table.shape[0]))
    print("Current Score: {},  Score_Std: {}".format(param_table.loc[best_param_index, "Score"],
                                                     param_table.loc[best_param_index, "Score_Std"]))
    print("--------------------------")
    for param in param_grid:
        best_param = param_table.loc[best_param_index, param]
        if isinstance(param_grid[param], list):
            if len(param_grid[param]) > 1 or (len(param_grid[param]) == 1 and param_grid[param][0] != best_param):
                print("{}: tuned to {}".format(param, best_param))
        else:
            print("{}: tuned to {}".format(param, best_param))
        param_grid[param] = [best_param]

    return param_grid, pred_return