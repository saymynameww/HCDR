import os
import numpy as np
import pandas as pd
from joblib import delayed
from sklearn.preprocessing import LabelEncoder
import gc

LABEL_COLUMN = 'TARGET'

DATA_DIR = '{}/data'.format(os.getcwd())
INPUT_FILE = os.path.join(DATA_DIR, 'application_train.csv.zip')
TEST_INPUT_FILE = os.path.join(DATA_DIR, 'application_test.csv.zip')
BUREAU_FILE = os.path.join(DATA_DIR, 'bureau.csv.zip')
BUREAU_BAL_FILE = os.path.join(DATA_DIR, 'bureau_balance.csv.zip')
PREV_APPLICATION_FILE = os.path.join(DATA_DIR, 'previous_application.csv.zip')
CREDIT_CARD_BAL_FILE = os.path.join(DATA_DIR, 'credit_card_balance.csv.zip')
POS_CASH_FILE = os.path.join(DATA_DIR, 'POS_CASH_balance.csv.zip')
INSTALLMENT_PAYMENT_FILE = os.path.join(DATA_DIR, 'installments_payments.csv.zip')
SUBMISSION_FILE = os.path.join(DATA_DIR, 'sample_submission.csv.zip')


def read_dataset():
    X = pd.read_csv(INPUT_FILE)
    X_test = pd.read_csv(TEST_INPUT_FILE)

    y = X[LABEL_COLUMN]
    del X['TARGET']

    bureau = pd.read_csv(BUREAU_FILE)
    bureau_bal = pd.read_csv(BUREAU_BAL_FILE)
    prev = pd.read_csv(PREV_APPLICATION_FILE)
    credit_card_bal = pd.read_csv(CREDIT_CARD_BAL_FILE)
    pos_cash = pd.read_csv(POS_CASH_FILE)
    installment_payment = pd.read_csv(INSTALLMENT_PAYMENT_FILE)

    # Convert categorical features
    categorical_features = [col for col in X.columns if X[col].dtype == 'object']

    train_test = pd.concat([X, X_test])
    train_test_one_hot = pd.get_dummies(train_test, columns=categorical_features)

    X = train_test_one_hot.iloc[:X.shape[0], :]
    X_test = train_test_one_hot.iloc[X.shape[0]:, ]

    return X, y, X_test, train_test, train_test_one_hot, bureau, bureau_bal, prev, credit_card_bal, pos_cash, installment_payment


def gen_relative_calculation(train_test_df):
    source_features = ['DAYS_EMPLOYED', 'DAYS_BIRTH', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'CNT_FAM_MEMBERS', 'AMT_ANNUITY', 'CNT_CHILDREN']
    application_df = train_test_df[['SK_ID_CURR'] + source_features].copy()

    application_df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    application_df['DAYS_EMPLOYED_PERC'] = application_df['DAYS_EMPLOYED'] / application_df['DAYS_BIRTH']
    application_df['INCOME_CREDIT_PERC'] = application_df['AMT_INCOME_TOTAL'] / application_df['AMT_CREDIT']
    application_df['INCOME_PER_PERSON'] = application_df['AMT_INCOME_TOTAL'] / application_df['CNT_FAM_MEMBERS']
    application_df['ANNUITY_INCOME_PERC'] = application_df['AMT_ANNUITY'] / application_df['AMT_INCOME_TOTAL']

    application_df['LOAN_INCOME_RATIO'] = application_df['AMT_CREDIT'] / application_df['AMT_INCOME_TOTAL']
    application_df['ANNUITY_LENGTH'] = application_df['AMT_CREDIT'] / application_df['AMT_ANNUITY']
    application_df['WORKING_LIFE_RATIO'] = application_df['DAYS_EMPLOYED'] / application_df['DAYS_BIRTH']
    application_df['INCOME_PER_FAM'] = application_df['AMT_INCOME_TOTAL'] / application_df['CNT_FAM_MEMBERS']
    application_df['CHILDREN_RATIO'] = application_df['CNT_CHILDREN'] / application_df['CNT_FAM_MEMBERS']

    application_df['PAYMENT_RATE'] = application_df['AMT_ANNUITY'] / application_df['AMT_CREDIT']

    for f in source_features:
        del application_df[f]

    return application_df


def gen_new_relative_feature(train_test_df):
    docs = [_f for _f in train_test_df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in train_test_df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

    source_features = ['DAYS_EMPLOYED', 'DAYS_BIRTH', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'CNT_FAM_MEMBERS',
                       'AMT_ANNUITY', 'CNT_CHILDREN', 'AMT_GOODS_PRICE', 'ORGANIZATION_TYPE', 'EXT_SOURCE_1',
                       'EXT_SOURCE_2', 'EXT_SOURCE_3', 'OWN_CAR_AGE', 'DAYS_LAST_PHONE_CHANGE']
    features = ['SK_ID_CURR'] + source_features + docs + live
    application_df = train_test_df[features].copy()

    inc_by_org = application_df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    df = application_df[['SK_ID_CURR']].copy()
    df['CREDIT_TO_GOODS_RATIO'] = application_df['AMT_CREDIT'] / application_df['AMT_GOODS_PRICE']
    df['DOC_IND_KURT'] = application_df[docs].kurtosis(axis=1)
    df['LIVE_IND_SUM'] = application_df[live].sum(axis=1)
    df['INCOME_PER_CHLD'] = application_df['AMT_INCOME_TOTAL'] / (1 + application_df['CNT_CHILDREN'])
    df['INCOME_BY_ORG'] = application_df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = application_df['AMT_ANNUITY'] / (1 + application_df['AMT_INCOME_TOTAL'])
    df['SOURCES_PRODUCT'] = application_df['EXT_SOURCE_1'] * application_df['EXT_SOURCE_2'] * application_df['EXT_SOURCE_3']
    df['EXT_SOURCES_MEAN'] = application_df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['EXT_SOURCES_STD'] = application_df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['EXT_SOURCES_STD'] = df['EXT_SOURCES_STD'].fillna(df['EXT_SOURCES_STD'].mean())
    df['CAR_TO_BIRTH_RATIO'] = application_df['OWN_CAR_AGE'] / application_df['DAYS_BIRTH']
    df['CAR_TO_EMPLOY_RATIO'] = application_df['OWN_CAR_AGE'] / application_df['DAYS_EMPLOYED']
    df['PHONE_TO_BIRTH_RATIO'] = application_df['DAYS_LAST_PHONE_CHANGE'] / application_df['DAYS_BIRTH']
    df['PHONE_TO_EMPLOY_RATIO'] = application_df['DAYS_LAST_PHONE_CHANGE'] / application_df['DAYS_EMPLOYED']

    return df


#
# Generated features from previous application
#


def gen_prev_installment_feature(installment_payment_df):
    sorted_prev = installment_payment_df[['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER']].sort_values(
        ['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'])
    compare_to_last = sorted_prev.groupby(by=['SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].diff()
    compare_to_last = compare_to_last.fillna(compare_to_last.mean())
    sorted_prev['COMPARED_TO_LAST'] = compare_to_last
    std_of_installment_seq = sorted_prev[['SK_ID_PREV', 'COMPARED_TO_LAST']].groupby(by=['SK_ID_PREV']).std().reset_index()
    std_of_installment_seq = std_of_installment_seq.rename(index=str,
                                                           columns={'COMPARED_TO_LAST': 'STD_OF_INSTALLMENT_SEQ'})
    prev_installment_feature = installment_payment_df[['SK_ID_CURR', 'SK_ID_PREV']].copy()
    prev_installment_feature = prev_installment_feature.merge(right=std_of_installment_seq, how='left', on='SK_ID_PREV')
    late_installment = installment_payment_df[
        ['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT']].sort_values(['SK_ID_CURR', 'SK_ID_PREV'])
    late_installment['LATE'] = late_installment['DAYS_INSTALMENT'] - late_installment['DAYS_ENTRY_PAYMENT']

    late_mean = late_installment[['SK_ID_PREV', 'LATE']].groupby(by=['SK_ID_PREV']).mean()
    late_mean = late_mean.fillna(late_mean.mean()).reset_index()
    late_mean = late_mean.rename(index=str, columns={'LATE': 'MEAN_OF_LATE_INSTALLMENT'})
    prev_installment_feature = prev_installment_feature.merge(right=late_mean, how='left', on='SK_ID_PREV')
    pay_less = installment_payment_df[['SK_ID_CURR', 'SK_ID_PREV', 'AMT_INSTALMENT', 'AMT_PAYMENT']].sort_values(
        ['SK_ID_CURR', 'SK_ID_PREV'])
    pay_less['INSUFFICIENT_PAYMENT'] = pay_less['AMT_INSTALMENT'] - pay_less['AMT_PAYMENT']
    pay_less = pay_less[['SK_ID_PREV', 'INSUFFICIENT_PAYMENT']].groupby(by=['SK_ID_PREV']).mean()
    pay_less = pay_less.fillna(pay_less.mean()).reset_index()
    pay_less = pay_less.rename(index=str, columns={'INSUFFICIENT_PAYMENT': 'MEAN_OF_INSUFFICIENT_PAYMENT'})
    prev_installment_feature = prev_installment_feature.merge(right=pay_less, how='left', on='SK_ID_PREV')
    prev_installment_feature_by_curr = prev_installment_feature.groupby(by=['SK_ID_CURR']).mean()
    prev_installment_feature_by_curr = prev_installment_feature_by_curr.fillna(prev_installment_feature_by_curr.mean()).reset_index()
    del prev_installment_feature_by_curr['SK_ID_PREV']

    assert 'SK_ID_CURR' in prev_installment_feature_by_curr.columns
    prev_installment_feature_by_curr = prev_installment_feature_by_curr.set_index('SK_ID_CURR')
    assert 'SK_ID_CURR' not in prev_installment_feature_by_curr.columns

    return prev_installment_feature_by_curr


#
# - bureau.csv
#     - All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample).
#     - For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application date.
#
# - bureau_balance.csv
#     - Monthly balances of previous credits in Credit Bureau.
#     - This table has one row for each month of history of every previous credit reported to Credit Bureau – i.e the table has (#loans in sample * # of relative previous credits * # of months where we have some history observable for the previous credits) rows.
#


def gen_bur_month_balance(bureau_df, bureau_bal_df):
    agg_by = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    bureau_bal_agg = bureau_bal_df.copy().groupby('SK_ID_BUREAU').agg(agg_by)
    bureau_bal_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bureau_bal_agg.columns.tolist()])

    bureau_agg = bureau_df.copy().join(bureau_bal_agg, how='left', on='SK_ID_BUREAU')
    bureau_agg = bureau_agg.reset_index()
    del bureau_agg['SK_ID_BUREAU']

    agg_by = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
        'AMT_ANNUITY': ['max', 'mean'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    del bureau_agg['CREDIT_ACTIVE'], bureau_agg['CREDIT_CURRENCY'], bureau_agg['CREDIT_TYPE']
    bureau_agg = bureau_agg.groupby('SK_ID_CURR').agg(agg_by)
    bureau_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    return bureau_agg.reset_index()


#
# credit_variety
#


def gen_credit_variety(bureau_df):
    count_day_credit = bureau_df[['SK_ID_CURR', 'DAYS_CREDIT']].groupby(by=['SK_ID_CURR'])['DAYS_CREDIT'].count()
    count_day_credit = count_day_credit.reset_index().rename(index=str, columns={'DAYS_CREDIT': 'BUREAU_LOAN_COUNT'})

    count_credit_type = bureau_df[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by=['SK_ID_CURR'])['CREDIT_TYPE'].nunique()
    count_credit_type = count_credit_type.reset_index().rename(index=str, columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'})

    credit_variety = count_day_credit.merge(right=count_credit_type, how='left', on='SK_ID_CURR')
    credit_variety['AVERAGE_LOAN_TYPE'] = credit_variety['BUREAU_LOAN_COUNT'] / credit_variety['BUREAU_LOAN_TYPES']

    return credit_variety


#
# bureau_active
#


def gen_bureau_active(bureau_df):
    def count_active(x):
        return 0 if x == 'Closed' else 1

    active = bureau_df[['SK_ID_CURR', 'CREDIT_ACTIVE']].apply(lambda x: count_active(x.CREDIT_ACTIVE), axis=1)
    bureau_active = bureau_df[['SK_ID_CURR']].copy()
    bureau_active['ACTIVE_COUNT'] = active
    bureau_active_sum = bureau_active.groupby(by=['SK_ID_CURR'])['ACTIVE_COUNT'].sum().reset_index()

    bureau_active_mean = bureau_active.groupby(by=['SK_ID_CURR'])['ACTIVE_COUNT'].mean().reset_index()
    bureau_active_mean = bureau_active_mean.rename(index=str, columns={'ACTIVE_COUNT': 'ACTIVE_LOANS_PERCENTAGE'})

    return bureau_active_sum.merge(right=bureau_active_mean, how='left', on='SK_ID_CURR')


#
# day_credit_group
#


def gen_day_credit_group(bureau_df):
    day_credit_group = bureau_df[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT']].groupby(by=['SK_ID_CURR'])
    day_credit_group = day_credit_group.apply(lambda x: x.sort_values(['DAYS_CREDIT'], ascending=False)).reset_index(
        drop=True)
    day_credit_group['DAYS_CREDIT1'] = day_credit_group['DAYS_CREDIT'] * -1
    day_credit_group['DAYS_DIFF'] = day_credit_group.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT1'].diff()
    day_credit_group['DAYS_DIFF'] = day_credit_group['DAYS_DIFF'].fillna(day_credit_group['DAYS_DIFF'].mean()).astype('uint32')
    del day_credit_group['DAYS_CREDIT1'], day_credit_group['DAYS_CREDIT'], day_credit_group['SK_ID_BUREAU']

    day_credit_group_mean = day_credit_group.groupby(by=['SK_ID_CURR'])['DAYS_DIFF'].mean()
    day_credit_group_mean = day_credit_group_mean.reset_index().rename(index=str, columns={'DAYS_DIFF': 'MEAN_DAYS_DIFF'})
    day_credit_group_max = day_credit_group.groupby(by=['SK_ID_CURR'])['DAYS_DIFF'].max()
    day_credit_group_max = day_credit_group_max.reset_index().rename(index=str, columns={'DAYS_DIFF': 'MAX_DAYS_DIFF'})

    return day_credit_group_mean.merge(right=day_credit_group_max, how='left', on='SK_ID_CURR')


#
# bureau_credit_time
#


def gen_bureau_credit_time(bureau_df):
    def check_credit_time(x):
        return 0 if x < 0 else 1

    credit_time = bureau_df[['SK_ID_CURR', 'DAYS_CREDIT_ENDDATE']].apply(lambda x: check_credit_time(x.DAYS_CREDIT_ENDDATE),
                                                                         axis=1)
    bureau_credit_time = bureau_df[['SK_ID_CURR']].copy()
    bureau_credit_time['CREDIT_TIME'] = credit_time

    credit_time_mean = bureau_credit_time.groupby(by=['SK_ID_CURR'])['CREDIT_TIME'].mean()
    credit_time_mean = credit_time_mean.reset_index().rename(index=str, columns={'CREDIT_TIME': 'MEAN_CREDIT_TIME'})

    credit_time_max = bureau_credit_time.groupby(by=['SK_ID_CURR'])['CREDIT_TIME'].max()
    credit_time_max = credit_time_max.reset_index().rename(index=str, columns={'CREDIT_TIME': 'MAX_CREDIT_TIME'})

    return credit_time_mean.merge(right=credit_time_max, how='left', on='SK_ID_CURR')


#
# loan_count
#


def gen_loan_count(bureau_df):
    positive_credit_end_date = bureau_df[bureau_df['DAYS_CREDIT_ENDDATE'] > 0]
    max_per_loan = positive_credit_end_date.groupby(by=['SK_ID_CURR', 'SK_ID_BUREAU'])[
        'DAYS_CREDIT_ENDDATE'].max().reset_index()
    max_per_loan = max_per_loan.rename(index=str, columns={'DAYS_CREDIT_ENDDATE': 'MAX_DAYS_CREDIT_ENDDATE'})
    max_per_user = max_per_loan.groupby(by=['SK_ID_CURR'])['MAX_DAYS_CREDIT_ENDDATE'].max().reset_index()

    current_loan_count = positive_credit_end_date.groupby(by=['SK_ID_CURR'])['SK_ID_BUREAU'].count().reset_index()
    current_loan_count = current_loan_count.rename(index=str, columns={'SK_ID_BUREAU': 'COUNT_SK_ID_BUREAU'})

    return max_per_user.merge(right=current_loan_count, how='left', on='SK_ID_CURR')


#
# cust_debt_to_credit
#


def gen_cust_debt_to_credit(bureau_df):
    bureau_df = bureau_df.copy()
    bureau_df['AMT_CREDIT_SUM_DEBT'] = bureau_df['AMT_CREDIT_SUM_DEBT'].fillna(bureau_df['AMT_CREDIT_SUM_DEBT'].mean())
    bureau_df['AMT_CREDIT_SUM'] = bureau_df['AMT_CREDIT_SUM'].fillna(bureau_df['AMT_CREDIT_SUM'].mean())
    cust_debt = bureau_df[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])[
        'AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
    cust_credit = bureau_df[['SK_ID_CURR', 'AMT_CREDIT_SUM']].groupby(by=['SK_ID_CURR'])[
        'AMT_CREDIT_SUM'].sum().reset_index().rename(index=str, columns={'AMT_CREDIT_SUM': 'TOTAL_CUSTOMER_CREDIT'})
    cust_profile = cust_debt.merge(cust_credit, on=['SK_ID_CURR'], how='left')
    cust_profile['DEBT_CREDIT_RATIO'] = cust_profile['TOTAL_CUSTOMER_DEBT'] / cust_profile['TOTAL_CUSTOMER_CREDIT']

    del cust_profile['TOTAL_CUSTOMER_DEBT'], cust_profile['TOTAL_CUSTOMER_CREDIT']
    assert len(list(cust_profile.columns)) == 2

    return cust_profile


#
# cust_overdue_debt
#


def gen_cust_overdue_debt(bureau_df):
    bureau_df = bureau_df.copy()
    bureau_df['AMT_CREDIT_SUM_DEBT'] = bureau_df['AMT_CREDIT_SUM_DEBT'].fillna(bureau_df['AMT_CREDIT_SUM_DEBT'].mean())
    bureau_df['AMT_CREDIT_SUM_OVERDUE'] = bureau_df['AMT_CREDIT_SUM_OVERDUE'].fillna(bureau_df['AMT_CREDIT_SUM_OVERDUE'].mean())
    cust_debt = bureau_df[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])[
        'AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
    cust_overdue = bureau_df[['SK_ID_CURR', 'AMT_CREDIT_SUM_OVERDUE']].groupby(by=['SK_ID_CURR'])[
        'AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename(index=str,
                                                             columns={'AMT_CREDIT_SUM_OVERDUE': 'TOTAL_CUSTOMER_OVERDUE'})
    cust_profile = cust_debt.merge(cust_overdue, on=['SK_ID_CURR'], how='left')
    cust_profile['OVERDUE_DEBT_RATIO'] = cust_profile['TOTAL_CUSTOMER_OVERDUE'] / cust_profile['TOTAL_CUSTOMER_DEBT']

    del cust_profile['TOTAL_CUSTOMER_OVERDUE'], cust_profile['TOTAL_CUSTOMER_DEBT']
    assert len(list(cust_profile.columns)) == 2

    return cust_profile


#
# avg_prolong
#


def gen_avg_prolong(bureau_df):
    bureau_df = bureau_df.copy()
    bureau_df['CNT_CREDIT_PROLONG'] = bureau_df['CNT_CREDIT_PROLONG'].fillna(bureau_df['CNT_CREDIT_PROLONG'].mean())
    avg_prolong = bureau_df[['SK_ID_CURR', 'CNT_CREDIT_PROLONG']].groupby(by=['SK_ID_CURR'])[
        'CNT_CREDIT_PROLONG'].mean().reset_index().rename(index=str,
                                                          columns={'CNT_CREDIT_PROLONG': 'AVG_CREDITDAYS_PROLONGED'})
    assert len(list(avg_prolong.columns)) == 2
    return avg_prolong


#
# avg_buro
#


def gen_avg_buro(bureau_df, bureau_bal_df):
    bureau_bal_df = bureau_bal_df.copy()
    buro_counts = bureau_bal_df.groupby('SK_ID_BUREAU')['STATUS'].value_counts(normalize=False)
    buro_counts_unstacked = buro_counts.unstack('STATUS')
    buro_counts_unstacked.columns = ['STATUS_0', 'STATUS_1', 'STATUS_2', 'STATUS_3', 'STATUS_4', 'STATUS_5', 'STATUS_C',
                                     'STATUS_X', ]

    bureau_df = bureau_df.copy().join(buro_counts_unstacked, how='left', on='SK_ID_BUREAU')
    buro_cat_features = [bcol for bcol in bureau_df.columns if bureau_df[bcol].dtype == 'object']
    bureau_df = pd.get_dummies(bureau_df, columns=buro_cat_features)

    avg_buro = bureau_df.groupby('SK_ID_CURR').mean()
    avg_buro.columns = ['SK_ID_BUREAU'] + ['MEAN_OF_{}'.format(c) for c in avg_buro.columns[1:]]
    avg_buro['BUREAU_COUNT'] = bureau_df[['SK_ID_BUREAU', 'SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']

    avg_buro = avg_buro.reset_index()
    assert 'SK_ID_CURR' in list(avg_buro.columns)

    del avg_buro['SK_ID_BUREAU']
    return avg_buro


# - POS_CASH_balance.csv
#     - Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
#     - This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) rows.

#
# max_pos_cash
# avg_pos_cash
# count_pos_cash
#


def gen_pos_cash_features(pos_cash_df):
    max_pos_cash = pos_cash_df[['SK_ID_CURR', 'MONTHS_BALANCE', 'CNT_INSTALMENT', 'SK_DPD', 'SK_DPD_DEF']].groupby(
        'SK_ID_CURR').max()
    max_pos_cash.columns = ['MAX_OF_{}'.format(c) for c in max_pos_cash.columns]
    avg_pos_cash = pos_cash_df[['SK_ID_CURR', 'MONTHS_BALANCE', 'CNT_INSTALMENT', 'SK_DPD', 'SK_DPD_DEF']].groupby(
        'SK_ID_CURR').mean()
    avg_pos_cash.columns = ['MEAN_OF_{}'.format(c) for c in avg_pos_cash.columns]
    count_pos_cash = pos_cash_df[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    count_pos_cash.columns = ['COUNT_OF_{}'.format(c) for c in count_pos_cash.columns]

    max_pos_cash = max_pos_cash.reset_index()
    avg_pos_cash = avg_pos_cash.reset_index()
    count_pos_cash = count_pos_cash.reset_index()

    return max_pos_cash.merge(right=avg_pos_cash, how='left', on='SK_ID_CURR').merge(right=count_pos_cash, how='left', on='SK_ID_CURR')


#
# pos_cash
#


def gen_agg_pos_cash(pos_cash_df):
    pos_cash_df = pos_cash_df.copy()

    agg_by = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }

    pos_cash_agg = pos_cash_df.copy().groupby('SK_ID_CURR').agg(agg_by)
    pos_cash_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in pos_cash_agg.columns.tolist()])
    return pos_cash_agg.reset_index()


def gen_mean_pos_cash(pos_cash_df):
    le = LabelEncoder()
    pos_cash_df['NAME_CONTRACT_STATUS'] = le.fit_transform(pos_cash_df['NAME_CONTRACT_STATUS'].astype(str))

    nunique_status = pos_cash_df[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
    nunique_status2 = pos_cash_df[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()

    pos_cash_df['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
    pos_cash_df['MAX_NUNIQUE_STATUS'] = nunique_status2['NAME_CONTRACT_STATUS']
    pos_cash_df.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

    mean_pos_cash = pos_cash_df.groupby('SK_ID_CURR').mean()
    mean_pos_cash.columns = ['MEAN_OF_{}'.format(c) for c in mean_pos_cash.columns]

    return mean_pos_cash.reset_index()


#
# avg_credit_card_bal
#

# - credit_card_balance.csv
#     - Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
#     - This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credit cards * # of months where we have some history observable for the previous credit card) rows.


def gen_avg_credit_card_bal(credit_card_bal_df):
    credit_card_cat_features = [col for col in credit_card_bal_df.columns if credit_card_bal_df[col].dtype == 'object']
    avg_credit_card_bal = credit_card_bal_df.copy().drop(credit_card_cat_features, axis=1).groupby('SK_ID_CURR').mean()
    del avg_credit_card_bal['SK_ID_PREV']
    avg_credit_card_bal.columns = ['MEAN_OF_{}'.format(c) for c in avg_credit_card_bal.columns]
    return avg_credit_card_bal.reset_index()


#
# credit_card_bal
#


def gen_agg_credit_card_bal(credit_card_bal_df):
    credit_card_bal_df = credit_card_bal_df.copy()
    credit_card_bal_agg = credit_card_bal_df.groupby('SK_ID_CURR').agg(['min', 'max', 'sum', 'var'])
    credit_card_bal_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in credit_card_bal_agg.columns.tolist()])
    credit_card_bal_agg['CC_COUNT'] = credit_card_bal_df.groupby('SK_ID_CURR').size()

    return credit_card_bal_agg.reset_index()


def gen_credit_card_bal(credit_card_bal_df):
    le = LabelEncoder()
    credit_card_bal_df['NAME_CONTRACT_STATUS'] = le.fit_transform(credit_card_bal_df['NAME_CONTRACT_STATUS'].astype(str))
    nunique_status = credit_card_bal_df[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
    nunique_status2 = credit_card_bal_df[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()

    credit_card_bal_df['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
    credit_card_bal_df['MAX_NUNIQUE_STATUS'] = nunique_status2['NAME_CONTRACT_STATUS']
    credit_card_bal_df.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

    avg_credit_card_bal = credit_card_bal_df.groupby('SK_ID_CURR').mean()
    avg_credit_card_bal.columns = ['MEAN_OF_{}'.format(c) for c in avg_credit_card_bal.columns]
    return avg_credit_card_bal.reset_index()


# - previous_application.csv
#     - All previous applications for Home Credit loans of clients who have loans in our sample.
#     - There is one row for each previous application related to loans in our data sample.

#
# avg_prev
#


def gen_agg_prev(prev_df):
    prev_df = prev_df.copy()
    prev_df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev_df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev_df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev_df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev_df['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    prev_df['APP_CREDIT_PERC'] = prev_df['AMT_APPLICATION'] / prev_df['AMT_CREDIT']

    agg_by = {
        'AMT_ANNUITY': ['min', 'max'],
        'AMT_APPLICATION': ['min', 'max'],
        'AMT_CREDIT': ['min', 'max'],
        'APP_CREDIT_PERC': ['min', 'max', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max'],
        'AMT_GOODS_PRICE': ['min', 'max'],
        'HOUR_APPR_PROCESS_START': ['min', 'max'],
        'RATE_DOWN_PAYMENT': ['min', 'max'],
        'DAYS_DECISION': ['min', 'max'],
        'CNT_PAYMENT': ['sum'],
    }

    prev_agg = prev_df.copy().groupby('SK_ID_CURR').agg(agg_by)
    prev_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    return prev_agg.reset_index()


def gen_avg_prev(prev_df):
    prev_cat_features = [pcol for pcol in prev_df.columns if prev_df[pcol].dtype == 'object']
    prev_df = pd.get_dummies(prev_df, columns=prev_cat_features)

    avg_prev = prev_df.groupby('SK_ID_CURR').mean()
    avg_prev.columns = ['SK_ID_PREV'] + ['MEAN_OF_{}'.format(c) for c in avg_prev.columns[1:]]
    cnt_prev = prev_df[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    avg_prev['COUNT_OF_SK_ID_PREV'] = cnt_prev['SK_ID_PREV']
    del avg_prev['SK_ID_PREV']

    avg_prev = avg_prev.reset_index()
    assert 'SK_ID_CURR' in list(avg_prev.columns)

    return avg_prev


# - installments_payments.csv
#     - Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.
#     - There is a) one row for every payment that was made plus b) one row each for missed payment.
#     - One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one previous Home Credit credit related to loans in our sample.

#
# avg_payments
#


def gen_agg_installments(installment_payment_df):
    installment_payment_agg = installment_payment_df.copy()

    installment_payment_agg['PAYMENT_PERC'] = installment_payment_agg['AMT_PAYMENT'] / installment_payment_agg['AMT_INSTALMENT']
    installment_payment_agg['PAYMENT_DIFF'] = installment_payment_agg['AMT_INSTALMENT'] - installment_payment_agg['AMT_PAYMENT']
    installment_payment_agg['DPD'] = installment_payment_agg['DAYS_ENTRY_PAYMENT'] - installment_payment_agg['DAYS_INSTALMENT']
    installment_payment_agg['DBD'] = installment_payment_agg['DAYS_INSTALMENT'] - installment_payment_agg['DAYS_ENTRY_PAYMENT']
    installment_payment_agg['DPD'] = installment_payment_agg['DPD'].apply(lambda x: x if x > 0 else 0)
    installment_payment_agg['DBD'] = installment_payment_agg['DBD'].apply(lambda x: x if x > 0 else 0)

    agg_by = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum', 'var'],
        'DBD': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum', 'var'],
        'AMT_PAYMENT': ['max', 'mean', 'sum', 'var'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum', 'var']
    }

    installment_payment_agg = installment_payment_agg.groupby('SK_ID_CURR').agg(agg_by)
    installment_payment_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in installment_payment_agg.columns.tolist()])
    return installment_payment_agg.reset_index()


def gen_avg_payments(installment_payment_df):
    installment_payment_df = installment_payment_df.copy()

    avg_payments = installment_payment_df.groupby('SK_ID_CURR').mean()
    del avg_payments['SK_ID_PREV']
    avg_payments.columns = ['MEAN_OF_{}'.format(c) for c in avg_payments.columns]

    avg_payments2 = installment_payment_df.groupby('SK_ID_CURR').max()
    del avg_payments2['SK_ID_PREV']
    avg_payments2.columns = ['MAX_OF_{}'.format(c) for c in avg_payments2.columns]

    avg_payments3 = installment_payment_df.groupby('SK_ID_CURR').min()
    del avg_payments3['SK_ID_PREV']
    avg_payments3.columns = ['MIN_OF_{}'.format(c) for c in avg_payments3.columns]

    avg_payments = avg_payments.reset_index()
    avg_payments2 = avg_payments2.reset_index()
    avg_payments3 = avg_payments3.reset_index()

    return avg_payments.merge(right=avg_payments2, how='left', on='SK_ID_CURR').merge(right=avg_payments3, how='left', on='SK_ID_CURR')


def get_feature_mapping(train_test, train_test_one_hot, bureau, bureau_bal, prev, credit_card_bal, pos_cash, installment_payment):
    return [
        (delayed(gen_relative_calculation)(train_test_one_hot)),
        (delayed(gen_new_relative_feature)(train_test)),
        (delayed(gen_prev_installment_feature)(installment_payment)),
        (delayed(gen_bur_month_balance)(bureau, bureau_bal)),
        (delayed(gen_credit_variety)(bureau)),
        (delayed(gen_bureau_active)(bureau)),
        (delayed(gen_day_credit_group)(bureau)),
        (delayed(gen_bureau_credit_time)(bureau)),
        (delayed(gen_loan_count)(bureau)),
        (delayed(gen_cust_debt_to_credit)(bureau)),
        (delayed(gen_cust_overdue_debt)(bureau)),
        (delayed(gen_avg_prolong)(bureau)),
        (delayed(gen_avg_buro)(bureau, bureau_bal)),
        (delayed(gen_pos_cash_features)(pos_cash)),
        (delayed(gen_agg_pos_cash)(pos_cash)),
        (delayed(gen_mean_pos_cash)(pos_cash)),
        (delayed(gen_avg_credit_card_bal)(credit_card_bal)),
        (delayed(gen_agg_credit_card_bal)(credit_card_bal)),
        (delayed(gen_credit_card_bal)(credit_card_bal)),
        (delayed(gen_agg_prev)(prev)),
        (delayed(gen_avg_prev)(prev)),
        (delayed(gen_agg_installments)(installment_payment)),
        (delayed(gen_avg_payments)(installment_payment)),
    ]


def columns_not_needed():
    return [
        'ACTIVE_CNT_CREDIT_PROLONG_SUM', 'ACTIVE_CREDIT_DAY_OVERDUE_MEAN', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_HOUR',
        'AMT_REQ_CREDIT_BUREAU_WEEK', 'BURO_CNT_CREDIT_PROLONG_SUM', 'BURO_CREDIT_ACTIVE_Bad debt_MEAN', 'BURO_CREDIT_ACTIVE_nan_MEAN',
        'BURO_CREDIT_CURRENCY_currency 1_MEAN', 'BURO_CREDIT_CURRENCY_currency 2_MEAN', 'BURO_CREDIT_CURRENCY_currency 3_MEAN',
        'BURO_CREDIT_CURRENCY_currency 4_MEAN', 'BURO_CREDIT_CURRENCY_nan_MEAN', 'BURO_CREDIT_DAY_OVERDUE_MAX', 'BURO_CREDIT_DAY_OVERDUE_MEAN',
        'BURO_CREDIT_TYPE_Cash loan (non-earmarked)_MEAN', 'BURO_CREDIT_TYPE_Interbank credit_MEAN', 'BURO_CREDIT_TYPE_Loan for business development_MEAN',
        'BURO_CREDIT_TYPE_Loan for purchase of shares (margin lending)_MEAN', 'BURO_CREDIT_TYPE_Loan for the purchase of equipment_MEAN',
        'BURO_CREDIT_TYPE_Loan for working capital replenishment_MEAN', 'BURO_CREDIT_TYPE_Mobile operator loan_MEAN',
        'BURO_CREDIT_TYPE_Real estate loan_MEAN', 'BURO_CREDIT_TYPE_Unknown type of loan_MEAN', 'BURO_CREDIT_TYPE_nan_MEAN',
        'BURO_MONTHS_BALANCE_MAX_MAX', 'BURO_STATUS_2_MEAN_MEAN', 'BURO_STATUS_3_MEAN_MEAN', 'BURO_STATUS_4_MEAN_MEAN', 'BURO_STATUS_5_MEAN_MEAN',
        'BURO_STATUS_nan_MEAN_MEAN', 'CC_AMT_DRAWINGS_ATM_CURRENT_MIN', 'CC_AMT_DRAWINGS_CURRENT_MIN', 'CC_AMT_DRAWINGS_OTHER_CURRENT_MAX',
        'CC_AMT_DRAWINGS_OTHER_CURRENT_MEAN', 'CC_AMT_DRAWINGS_OTHER_CURRENT_MIN', 'CC_AMT_DRAWINGS_OTHER_CURRENT_SUM',
        'CC_AMT_DRAWINGS_OTHER_CURRENT_VAR', 'CC_AMT_INST_MIN_REGULARITY_MIN', 'CC_AMT_PAYMENT_TOTAL_CURRENT_MIN', 'CC_AMT_PAYMENT_TOTAL_CURRENT_VAR',
        'CC_AMT_RECIVABLE_SUM', 'CC_AMT_TOTAL_RECEIVABLE_MAX', 'CC_AMT_TOTAL_RECEIVABLE_MIN', 'CC_AMT_TOTAL_RECEIVABLE_SUM', 'CC_AMT_TOTAL_RECEIVABLE_VAR',
        'CC_CNT_DRAWINGS_ATM_CURRENT_MIN', 'CC_CNT_DRAWINGS_CURRENT_MIN', 'CC_CNT_DRAWINGS_OTHER_CURRENT_MAX', 'CC_CNT_DRAWINGS_OTHER_CURRENT_MEAN',
        'CC_CNT_DRAWINGS_OTHER_CURRENT_MIN', 'CC_CNT_DRAWINGS_OTHER_CURRENT_SUM', 'CC_CNT_DRAWINGS_OTHER_CURRENT_VAR', 'CC_CNT_DRAWINGS_POS_CURRENT_SUM',
        'CC_CNT_INSTALMENT_MATURE_CUM_MAX', 'CC_CNT_INSTALMENT_MATURE_CUM_MIN', 'CC_COUNT', 'CC_MONTHS_BALANCE_MAX', 'CC_MONTHS_BALANCE_MEAN',
        'CC_MONTHS_BALANCE_MIN', 'CC_MONTHS_BALANCE_SUM', 'CC_NAME_CONTRACT_STATUS_Active_MAX', 'CC_NAME_CONTRACT_STATUS_Active_MIN',
        'CC_NAME_CONTRACT_STATUS_Approved_MAX', 'CC_NAME_CONTRACT_STATUS_Approved_MEAN', 'CC_NAME_CONTRACT_STATUS_Approved_MIN',
        'CC_NAME_CONTRACT_STATUS_Approved_SUM', 'CC_NAME_CONTRACT_STATUS_Approved_VAR', 'CC_NAME_CONTRACT_STATUS_Completed_MAX',
        'CC_NAME_CONTRACT_STATUS_Completed_MEAN', 'CC_NAME_CONTRACT_STATUS_Completed_MIN', 'CC_NAME_CONTRACT_STATUS_Completed_SUM', 'CC_NAME_CONTRACT_STATUS_Completed_VAR',
        'CC_NAME_CONTRACT_STATUS_Demand_MAX', 'CC_NAME_CONTRACT_STATUS_Demand_MEAN', 'CC_NAME_CONTRACT_STATUS_Demand_MIN', 'CC_NAME_CONTRACT_STATUS_Demand_SUM',
        'CC_NAME_CONTRACT_STATUS_Demand_VAR', 'CC_NAME_CONTRACT_STATUS_Refused_MAX', 'CC_NAME_CONTRACT_STATUS_Refused_MEAN', 'CC_NAME_CONTRACT_STATUS_Refused_MIN',
        'CC_NAME_CONTRACT_STATUS_Refused_SUM', 'CC_NAME_CONTRACT_STATUS_Refused_VAR', 'CC_NAME_CONTRACT_STATUS_Sent proposal_MAX',
        'CC_NAME_CONTRACT_STATUS_Sent proposal_MEAN', 'CC_NAME_CONTRACT_STATUS_Sent proposal_MIN', 'CC_NAME_CONTRACT_STATUS_Sent proposal_SUM',
        'CC_NAME_CONTRACT_STATUS_Sent proposal_VAR', 'CC_NAME_CONTRACT_STATUS_Signed_MAX', 'CC_NAME_CONTRACT_STATUS_Signed_MEAN', 'CC_NAME_CONTRACT_STATUS_Signed_MIN',
        'CC_NAME_CONTRACT_STATUS_Signed_SUM', 'CC_NAME_CONTRACT_STATUS_Signed_VAR', 'CC_NAME_CONTRACT_STATUS_nan_MAX', 'CC_NAME_CONTRACT_STATUS_nan_MEAN',
        'CC_NAME_CONTRACT_STATUS_nan_MIN', 'CC_NAME_CONTRACT_STATUS_nan_SUM', 'CC_NAME_CONTRACT_STATUS_nan_VAR', 'CC_SK_DPD_DEF_MAX',
        'CC_SK_DPD_DEF_MIN', 'CC_SK_DPD_DEF_SUM', 'CC_SK_DPD_DEF_VAR', 'CC_SK_DPD_MAX', 'CC_SK_DPD_MEAN', 'CC_SK_DPD_MIN', 'CC_SK_DPD_SUM',
        'CC_SK_DPD_VAR', 'CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN', 'CLOSED_AMT_CREDIT_SUM_LIMIT_SUM', 'CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN',
        'CLOSED_CNT_CREDIT_PROLONG_SUM', 'CLOSED_CREDIT_DAY_OVERDUE_MAX', 'CLOSED_CREDIT_DAY_OVERDUE_MEAN', 'CLOSED_MONTHS_BALANCE_MAX_MAX',
        'CNT_CHILDREN', 'ELEVATORS_MEDI', 'ELEVATORS_MODE', 'EMERGENCYSTATE_MODE_No', 'EMERGENCYSTATE_MODE_Yes', 'ENTRANCES_MODE', 'FLAG_CONT_MOBILE',
        'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
        'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
        'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_9', 'FLAG_EMAIL', 'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_OWN_CAR', 'FLOORSMAX_MODE',
        'FONDKAPREMONT_MODE_not specified', 'FONDKAPREMONT_MODE_org spec account', 'FONDKAPREMONT_MODE_reg oper account', 'FONDKAPREMONT_MODE_reg oper spec account',
        'HOUSETYPE_MODE_block of flats', 'HOUSETYPE_MODE_specific housing', 'HOUSETYPE_MODE_terraced house', 'LIVE_REGION_NOT_WORK_REGION',
        'NAME_CONTRACT_TYPE_Revolving loans', 'NAME_EDUCATION_TYPE_Academic degree', 'NAME_FAMILY_STATUS_Civil marriage', 'NAME_FAMILY_STATUS_Single / not married',
        'NAME_FAMILY_STATUS_Unknown', 'NAME_FAMILY_STATUS_Widow', 'NAME_HOUSING_TYPE_Co-op apartment', 'NAME_HOUSING_TYPE_With parents',
        'NAME_INCOME_TYPE_Businessman', 'NAME_INCOME_TYPE_Maternity leave', 'NAME_INCOME_TYPE_Pensioner', 'NAME_INCOME_TYPE_Student',
        'NAME_INCOME_TYPE_Unemployed', 'NAME_TYPE_SUITE_Children', 'NAME_TYPE_SUITE_Family', 'NAME_TYPE_SUITE_Group of people',
        'NAME_TYPE_SUITE_Other_A', 'NAME_TYPE_SUITE_Other_B', 'NAME_TYPE_SUITE_Spouse, partner', 'NAME_TYPE_SUITE_Unaccompanied',
        'NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_MEAN', 'NEW_RATIO_BURO_AMT_CREDIT_SUM_LIMIT_SUM', 'NEW_RATIO_BURO_AMT_CREDIT_SUM_OVERDUE_MEAN',
        'NEW_RATIO_BURO_CNT_CREDIT_PROLONG_SUM', 'NEW_RATIO_BURO_CREDIT_DAY_OVERDUE_MAX', 'NEW_RATIO_BURO_CREDIT_DAY_OVERDUE_MEAN', 'NEW_RATIO_BURO_MONTHS_BALANCE_MAX_MAX',
        'NEW_RATIO_PREV_AMT_DOWN_PAYMENT_MIN', 'NEW_RATIO_PREV_RATE_DOWN_PAYMENT_MAX', 'OCCUPATION_TYPE_Cleaning staff', 'OCCUPATION_TYPE_Cooking staff',
        'OCCUPATION_TYPE_HR staff', 'OCCUPATION_TYPE_IT staff', 'OCCUPATION_TYPE_Low-skill Laborers', 'OCCUPATION_TYPE_Managers',
        'OCCUPATION_TYPE_Private service staff', 'OCCUPATION_TYPE_Realty agents', 'OCCUPATION_TYPE_Sales staff', 'OCCUPATION_TYPE_Secretaries',
        'OCCUPATION_TYPE_Security staff', 'OCCUPATION_TYPE_Waiters/barmen staff', 'ORGANIZATION_TYPE_Advertising', 'ORGANIZATION_TYPE_Agriculture',
        'ORGANIZATION_TYPE_Business Entity Type 1', 'ORGANIZATION_TYPE_Business Entity Type 2', 'ORGANIZATION_TYPE_Cleaning', 'ORGANIZATION_TYPE_Culture',
        'ORGANIZATION_TYPE_Electricity', 'ORGANIZATION_TYPE_Emergency', 'ORGANIZATION_TYPE_Government', 'ORGANIZATION_TYPE_Hotel', 'ORGANIZATION_TYPE_Housing',
        'ORGANIZATION_TYPE_Industry: type 1', 'ORGANIZATION_TYPE_Industry: type 10', 'ORGANIZATION_TYPE_Industry: type 11', 'ORGANIZATION_TYPE_Industry: type 12',
        'ORGANIZATION_TYPE_Industry: type 13', 'ORGANIZATION_TYPE_Industry: type 2', 'ORGANIZATION_TYPE_Industry: type 3', 'ORGANIZATION_TYPE_Industry: type 4',
        'ORGANIZATION_TYPE_Industry: type 5', 'ORGANIZATION_TYPE_Industry: type 6', 'ORGANIZATION_TYPE_Industry: type 7', 'ORGANIZATION_TYPE_Industry: type 8',
        'ORGANIZATION_TYPE_Insurance', 'ORGANIZATION_TYPE_Legal Services', 'ORGANIZATION_TYPE_Mobile', 'ORGANIZATION_TYPE_Other', 'ORGANIZATION_TYPE_Postal',
        'ORGANIZATION_TYPE_Realtor', 'ORGANIZATION_TYPE_Religion', 'ORGANIZATION_TYPE_Restaurant', 'ORGANIZATION_TYPE_Security',
        'ORGANIZATION_TYPE_Security Ministries', 'ORGANIZATION_TYPE_Services', 'ORGANIZATION_TYPE_Telecom', 'ORGANIZATION_TYPE_Trade: type 1',
        'ORGANIZATION_TYPE_Trade: type 2', 'ORGANIZATION_TYPE_Trade: type 3', 'ORGANIZATION_TYPE_Trade: type 4', 'ORGANIZATION_TYPE_Trade: type 5',
        'ORGANIZATION_TYPE_Trade: type 6', 'ORGANIZATION_TYPE_Trade: type 7',
        'ORGANIZATION_TYPE_Transport: type 1', 'ORGANIZATION_TYPE_Transport: type 2', 'ORGANIZATION_TYPE_Transport: type 4', 'ORGANIZATION_TYPE_University',
        'ORGANIZATION_TYPE_XNA', 'POS_NAME_CONTRACT_STATUS_Amortized debt_MEAN', 'POS_NAME_CONTRACT_STATUS_Approved_MEAN', 'POS_NAME_CONTRACT_STATUS_Canceled_MEAN',
        'POS_NAME_CONTRACT_STATUS_Demand_MEAN', 'POS_NAME_CONTRACT_STATUS_XNA_MEAN', 'POS_NAME_CONTRACT_STATUS_nan_MEAN', 'PREV_CHANNEL_TYPE_Car dealer_MEAN',
        'PREV_CHANNEL_TYPE_nan_MEAN', 'PREV_CODE_REJECT_REASON_CLIENT_MEAN', 'PREV_CODE_REJECT_REASON_SYSTEM_MEAN', 'PREV_CODE_REJECT_REASON_VERIF_MEAN',
        'PREV_CODE_REJECT_REASON_XNA_MEAN', 'PREV_CODE_REJECT_REASON_nan_MEAN', 'PREV_FLAG_LAST_APPL_PER_CONTRACT_N_MEAN', 'PREV_FLAG_LAST_APPL_PER_CONTRACT_Y_MEAN',
        'PREV_FLAG_LAST_APPL_PER_CONTRACT_nan_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Building a house or an annex_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Business development_MEAN',
        'PREV_NAME_CASH_LOAN_PURPOSE_Buying a garage_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Buying a home_MEAN',
        'PREV_NAME_CASH_LOAN_PURPOSE_Buying a new car_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Buying a used car_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Education_MEAN',
        'PREV_NAME_CASH_LOAN_PURPOSE_Everyday expenses_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Furniture_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Gasification / water supply_MEAN',
        'PREV_NAME_CASH_LOAN_PURPOSE_Hobby_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Journey_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Money for a third person_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Other_MEAN',
        'PREV_NAME_CASH_LOAN_PURPOSE_Payments on other loans_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Refusal to name the goal_MEAN',
        'PREV_NAME_CASH_LOAN_PURPOSE_Wedding / gift / holiday_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_XAP_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_nan_MEAN', 'PREV_NAME_CLIENT_TYPE_XNA_MEAN',
        'PREV_NAME_CLIENT_TYPE_nan_MEAN', 'PREV_NAME_CONTRACT_STATUS_Unused offer_MEAN', 'PREV_NAME_CONTRACT_STATUS_nan_MEAN', 'PREV_NAME_CONTRACT_TYPE_XNA_MEAN',
        'PREV_NAME_CONTRACT_TYPE_nan_MEAN', 'PREV_NAME_GOODS_CATEGORY_Additional Service_MEAN', 'PREV_NAME_GOODS_CATEGORY_Animals_MEAN',
        'PREV_NAME_GOODS_CATEGORY_Auto Accessories_MEAN', 'PREV_NAME_GOODS_CATEGORY_Clothing and Accessories_MEAN', 'PREV_NAME_GOODS_CATEGORY_Construction Materials_MEAN',
        'PREV_NAME_GOODS_CATEGORY_Direct Sales_MEAN', 'PREV_NAME_GOODS_CATEGORY_Education_MEAN', 'PREV_NAME_GOODS_CATEGORY_Fitness_MEAN',
        'PREV_NAME_GOODS_CATEGORY_Gardening_MEAN', 'PREV_NAME_GOODS_CATEGORY_Homewares_MEAN', 'PREV_NAME_GOODS_CATEGORY_House Construction_MEAN',
        'PREV_NAME_GOODS_CATEGORY_Insurance_MEAN', 'PREV_NAME_GOODS_CATEGORY_Jewelry_MEAN', 'PREV_NAME_GOODS_CATEGORY_Medical Supplies_MEAN',
        'PREV_NAME_GOODS_CATEGORY_Medicine_MEAN', 'PREV_NAME_GOODS_CATEGORY_Office Appliances_MEAN', 'PREV_NAME_GOODS_CATEGORY_Other_MEAN', 'PREV_NAME_GOODS_CATEGORY_Tourism_MEAN',
        'PREV_NAME_GOODS_CATEGORY_Vehicles_MEAN', 'PREV_NAME_GOODS_CATEGORY_Weapon_MEAN', 'PREV_NAME_GOODS_CATEGORY_XNA_MEAN', 'PREV_NAME_GOODS_CATEGORY_nan_MEAN',
        'PREV_NAME_PAYMENT_TYPE_Cashless from the account of the employer_MEAN', 'PREV_NAME_PAYMENT_TYPE_Non-cash from your account_MEAN', 'PREV_NAME_PAYMENT_TYPE_nan_MEAN',
        'PREV_NAME_PORTFOLIO_Cars_MEAN', 'PREV_NAME_PORTFOLIO_nan_MEAN', 'PREV_NAME_PRODUCT_TYPE_nan_MEAN', 'PREV_NAME_SELLER_INDUSTRY_Construction_MEAN',
        'PREV_NAME_SELLER_INDUSTRY_Furniture_MEAN', 'PREV_NAME_SELLER_INDUSTRY_Industry_MEAN', 'PREV_NAME_SELLER_INDUSTRY_Jewelry_MEAN', 'PREV_NAME_SELLER_INDUSTRY_MLM partners_MEAN',
        'PREV_NAME_SELLER_INDUSTRY_Tourism_MEAN', 'PREV_NAME_SELLER_INDUSTRY_nan_MEAN', 'PREV_NAME_TYPE_SUITE_Group of people_MEAN', 'PREV_NAME_YIELD_GROUP_nan_MEAN',
        'PREV_PRODUCT_COMBINATION_POS industry without interest_MEAN', 'PREV_PRODUCT_COMBINATION_POS mobile without interest_MEAN', 'PREV_PRODUCT_COMBINATION_POS others without interest_MEAN',
        'PREV_PRODUCT_COMBINATION_nan_MEAN', 'PREV_WEEKDAY_APPR_PROCESS_START_nan_MEAN', 'REFUSED_AMT_DOWN_PAYMENT_MAX', 'REFUSED_AMT_DOWN_PAYMENT_MEAN',
        'REFUSED_RATE_DOWN_PAYMENT_MIN', 'REG_CITY_NOT_WORK_CITY', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
        'WALLSMATERIAL_MODE_Block', 'WALLSMATERIAL_MODE_Mixed', 'WALLSMATERIAL_MODE_Monolithic', 'WALLSMATERIAL_MODE_Others', 'WALLSMATERIAL_MODE_Panel',
        'WALLSMATERIAL_MODE_Wooden', 'WEEKDAY_APPR_PROCESS_START_FRIDAY', 'WEEKDAY_APPR_PROCESS_START_THURSDAY', 'WEEKDAY_APPR_PROCESS_START_TUESDAY'
    ]


def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def application_train_test():
    df = get_train_test_df()
    df = df[df['CODE_GENDER'] != 'XNA']

    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_DOC_IND_AVG'] = df[docs].mean(axis=1)
    df['NEW_DOC_IND_STD'] = df[docs].std(axis=1)
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_LIVE_IND_STD'] = df[live].std(axis=1)
    df['NEW_LIVE_IND_KURT'] = df[live].kurtosis(axis=1)
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_EMPLOY_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    df, cat_cols = one_hot_encoder(df, nan_as_category=False)
    df.drop(columns_not_needed(), axis=1, inplace=True, errors='ignore')

    return df


def get_train_test_df():
    df = pd.read_csv(INPUT_FILE)
    test_df = pd.read_csv(TEST_INPUT_FILE)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    return df


def bureau_and_balance():
    bureau = pd.read_csv(BUREAU_FILE)
    bb = pd.read_csv(BUREAU_BAL_FILE)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category=True)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category=True)

    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }

    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    cols = active_agg.columns.tolist()
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()

    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')

    for e in cols:
        bureau_agg['NEW_RATIO_BURO_' + e[0] + "_" + e[1].upper()] = bureau_agg['ACTIVE_' + e[0] + "_" + e[1].upper()] / bureau_agg['CLOSED_' + e[0] + "_" + e[1].upper()]

    bureau_agg.drop(columns_not_needed(), axis=1, inplace=True, errors='ignore')

    return bureau_agg


def previous_applications():
    prev = pd.read_csv(PREV_APPLICATION_FILE)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)

    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }

    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    cols = approved_agg.columns.tolist()
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev

    for e in cols:
        prev_agg['NEW_RATIO_PREV_' + e[0] + "_" + e[1].upper()] = prev_agg['APPROVED_' + e[0] + "_" + e[1].upper()] / prev_agg['REFUSED_' + e[0] + "_" + e[1].upper()]

    prev_agg.drop(columns_not_needed(), axis=1, inplace=True, errors='ignore')

    return prev_agg


def pos_cash():
    pos = pd.read_csv(POS_CASH_FILE)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)

    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()

    pos_agg.drop(columns_not_needed(), axis=1, inplace=True, errors='ignore')

    return pos_agg


def installments_payments():
    ins = pd.read_csv(INSTALLMENT_PAYMENT_FILE)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)

    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']

    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()

    ins_agg.drop(columns_not_needed(), axis=1, inplace=True, errors='ignore')

    return ins_agg


def credit_card_balance():
    cc = pd.read_csv(CREDIT_CARD_BAL_FILE)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)

    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()

    cc_agg.drop(columns_not_needed(), axis=1, inplace=True, errors='ignore')

    return cc_agg


def gen_reduced_relative_calculation():
    application_df = get_train_test_df()
    df = application_df[['SK_ID_CURR']].copy()

    application_df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df['INCOME_CREDIT_PERC'] = application_df['AMT_INCOME_TOTAL'] / application_df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = application_df['AMT_INCOME_TOTAL'] / application_df['CNT_FAM_MEMBERS']
    df['ANNUITY_LENGTH'] = application_df['AMT_CREDIT'] / application_df['AMT_ANNUITY']
    df['CHILDREN_RATIO'] = application_df['CNT_CHILDREN'] / application_df['CNT_FAM_MEMBERS']

    df = df.set_index('SK_ID_CURR')
    assert 'SK_ID_CURR' not in df.columns
    return df


def get_selected_features_df():
    return [
        (delayed(application_train_test)()),
        (delayed(bureau_and_balance)()),
        (delayed(previous_applications)()),
        (delayed(pos_cash)()),
        (delayed(installments_payments)()),
        (delayed(credit_card_balance)()),
        (delayed(gen_reduced_relative_calculation)()),
        (delayed(gen_prev_installment_feature)(pd.read_csv(INSTALLMENT_PAYMENT_FILE))),
    ]