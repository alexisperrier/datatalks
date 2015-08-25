#!/usr/bin/python
# Filename: fct.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn import metrics, cross_validation
from sklearn.metrics import make_scorer
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from scipy import stats

def check_shape(df_dict):
    for key, val  in df_dict.items():
        print("df %s: %s" % (key, val.shape))


def dumify(train, FX_test, k_test, features):
    train['is_train']   = 1
    FX_test['is_train'] = 2
    k_test['is_train']  = 3
    drop_last = True
    drop_original = True
    df = pd.concat([train, FX_test, k_test])

    for var in features:
        if var in df.columns:
            dummy_df = pd.get_dummies(df[var]).rename(columns=lambda x: var + '_' + str(x))
            if drop_last:
                last_col = dummy_df.columns[dummy_df.columns.size -1]
                dummy_df.drop(last_col, axis=1, inplace=True)
            df = pd.concat([df, dummy_df], axis=1)
            if drop_original:
                df.drop(var, axis=1, inplace=True)

    train   = df[df['is_train'] == 1]
    FX_test = df[df['is_train'] == 2]
    k_test  = df[df['is_train'] == 3]

    train.drop('is_train', axis=1, inplace=True)
    FX_test.drop('is_train', axis=1, inplace=True)
    k_test.drop('is_train', axis=1, inplace=True)

    print("\nafter dumify")
    print("train shape: " + str(train.shape))
    print("FX_test shape: " + str(FX_test.shape))
    print("k_test shape: " + str(k_test.shape))
    print("train: found Nan values: " + str(train[train.isnull().any(axis=1)].shape[0]))
    print("FX_test: found Nan values: " + str(FX_test[FX_test.isnull().any(axis=1)].shape[0]))
    print("k_test: found Nan values: " + str(k_test[k_test.isnull().any(axis=1)].shape[0]))
    return train, FX_test, k_test


def subset(raw_train, raw_y, n_samples):
    train_idx  = np.array(random.sample(list(raw_y.index), n_samples))
    test_idx   = np.setdiff1d(np.array(raw_y.index), np.array(train_idx)) # this could be limited to n_samples for better perfs
    y          = raw_y[train_idx]
    Fy_test    = raw_y[test_idx]
    train      = raw_train.loc[y.index]
    FX_test    = raw_train.loc[Fy_test.index]

    check_shape({'raw_y': raw_y, 'raw_train':raw_train })
    check_shape({'train_idx': train_idx, 'test_idx':test_idx })
    check_shape({'y': y, 'train': train,'Fy_test':Fy_test, 'FX_test': FX_test })
    return train, y, FX_test, Fy_test

def load_data():
    train = pd.read_csv('../data/train.csv', header=0)        # Load the train file into a dataframe
    test  = pd.read_csv('../data/test.csv',  header=0)        # Load the test file into a dataframe
    return (train, test)

def health(train, test):
    # print("todo: check for Near Zero Variance, Outliers, Skewness, ")
    if train[train.isnull().any(axis=1)].shape[0] > 1:
        print("found " + str(train[train.isnull().any(axis=1)].shape[0]) + " Nan Values in train")
    if test[test.isnull().any(axis=1)].shape[0] > 1:
        print("found " + str(test[test.isnull().any(axis=1)].shape[0]) + " Nan Values in test")

def plot_deviance(est, X_test, y_test, ax=None, label='', train_color='#2c7bb6', test_color='#d7191c', alpha=1.0):
    n_estimators = len(est.estimators_)
    test_dev = np.empty(n_estimators)

    for i, pred in enumerate(est.staged_predict(X_test)):
       test_dev[i] = est.loss_(y_test, pred)
    plt.figure()
    plt.title('Deviance plot: error as function of N Trees')
    plt.plot(np.arange(n_estimators) + 1, test_dev, color=test_color, label='Test', linewidth=2, alpha=alpha)
    plt.plot(np.arange(n_estimators) + 1, est.train_score_, color=train_color, label='Train', linewidth=2, alpha=alpha)
    plt.ylabel('Error')
    plt.xlabel('n_estimators')
    plt.legend(loc="best")
    plt.show(block= False)
    return test_dev


def plot_learning_curve(train_sizes, train_scores, test_scores):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)
    plt.figure()
    plt.title('Learning Curve')
    plt.xlabel("Training examples")
    plt.ylabel("Gini Score")
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="b")
    plt.plot(train_sizes, train_scores_mean, 'x-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean,  'o-', color="b", label="Cross-validation score")
    plt.legend(loc="best", title="Gamma")
    plt.show(block= False)

# scoring function
# see http://stats.stackexchange.com/questions/110599/how-to-get-both-mse-and-r2-from-a-sklearn-gridsearchcv
def gini(expected, predicted):
    _EXPECTED  = 0
    _PREDICTED = 1
    _INDEX     = 2
    assert expected.shape[0] == predicted.shape[0], 'unequal number of rows'
    _all = np.asarray(np.c_[ expected, predicted, np.arange(expected.shape[0])], dtype=np.float)
    # sort by predicted descending, then by index ascending
    sort_order = np.lexsort((_all[:, _INDEX], -1 * _all[:, _PREDICTED]))
    _all = _all[sort_order]
    total_losses = _all[:, _EXPECTED].sum()
    gini_sum     = _all[:, _EXPECTED].cumsum().sum() / total_losses
    gini_sum    -= (expected.shape[0] + 1.0) / 2.0
    return gini_sum / expected.shape[0]

def gini_normalized(expected, predicted, verbose = False):
    gini_norm = gini(expected, predicted) / gini(expected, expected)
    if verbose:
        print('Gini: %2.3f' % gini_norm)
    return gini_norm

def gini_score(y_true,y_pred):
    # MSE(y_true,y_pred) #set score here and not below if using MSE in GridCV
    # score = R2(y_true,y_pred)
    score = gini_normalized(y_true,y_pred)
    return score

def gini_scorer():
    return make_scorer(gini_score, greater_is_better=True)

def gini_loss():
    return make_scorer(gini_score, greater_is_better=False)

