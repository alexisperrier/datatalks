'''
This script is a submission for the Kaggle Libert Mutual Competition
https://www.kaggle.com/c/liberty-mutual-group-property-inspection-prediction
It's an exercise in stacking several models.
The first layer obtains predictions for standard algorithms:
For instance: Gradient Boosting, Support Vector Regression, Stochastic Gradient, ...
The predictions of the first layer are fed to a simple linear regression
that constitutes the second layer.
The different predictions are also weighted with the gini score obtained during
the cross validation phase
The metric used for final validation is the Gini Coefficient metric
This script did not result in a good score. It's a first step toward better
stacking and blending architectures and optimization

Author: Alex Perrier alexis.perrier@gmail.com
'''

import fct as gal_fct
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from scipy import stats
from sklearn.cross_validation   import StratifiedKFold, KFold
from sklearn.ensemble           import GradientBoostingRegressor, RandomForestRegressor
from sklearn.learning_curve     import learning_curve
from sklearn.linear_model       import LinearRegression, SGDRegressor
from sklearn.metrics            import r2_score
from sklearn.neighbors          import KNeighborsRegressor
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.svm                import SVR

pd.options.mode.chained_assignment = None

experiment = {
    'n_folds':       10,
    'models':        [
        GradientBoostingRegressor(n_estimators= 5000,
                                min_samples_leaf=20,
                                max_features = 1.0,
                                learning_rate= 0.001,
                                max_depth= 5,
                                loss='ls'),
        GradientBoostingRegressor(n_estimators= 5000,
                                min_samples_leaf=20,
                                max_features = 0.5,
                                learning_rate= 0.001,
                                max_depth= 2,
                                loss='ls'),
        SVR(kernel = 'rbf',  C=1.0,  epsilon=0.1, degree=2, gamma=0.005),
        SVR(kernel = 'poly', C=0.5,  epsilon=0.1, degree=3, gamma=0.01),
        SGDRegressor(loss='huber', penalty= 'l2', alpha= 0.01, epsilon=0.1,
                    learning_rate='constant', eta0=  0.01),
    ],
}
experiment['n_models'] = len(experiment['models'])

# Defines what steps will be carried out
steps= {
    'cross_validation':  False,
    'show_figure':       True,
    'grid_search':       True,
    'learning curve':    True,
    'final_validation':  True,
    'kaggle_submission': True,
    'deviance_plot':     True,        # only for gradient boosting
}
# Subsetting the features
features = {
    'nzv':     ['T1_V7', 'T1_V8','T1_V12', 'T1_V15', 'T2_V8'],
    'cat':     ['T1_V4', 'T1_V5', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11','T1_V16',
                'T1_V12', 'T1_V15','T2_V5','T2_V13'],
    'boolean': ['T1_V6','T1_V17','T2_V3','T2_V11', 'T2_V12'],
    'cont':    ['T1_V1','T1_V2','T1_V3','T1_V10','T1_V13','T1_V14','T2_V1',
                'T2_V2','T2_V4','T2_V6','T2_V7','T2_V8','T2_V9','T2_V10',
                'T2_V14','T2_V15'],
    'low':     ['T2_V10', 'T2_V11', 'T1_V13', 'T1_V10']  # low importance features
}

np.random.seed(0) # seed to shuffle the train set
raw_train = pd.read_csv('../data/train.csv', index_col=0)
raw_test  = pd.read_csv('../data/test.csv', index_col=0)
raw_y     = raw_train['Hazard']
raw_train.drop('Hazard', axis=1, inplace=True)
raw_test_index = raw_test.index

# Transform the data
X   = np.array(raw_train)
XX  = np.array(raw_test)
y   = np.array(raw_y)

for i in range(X.shape[1]):
    lbl     = LabelEncoder()
    lbl.fit(list(X[:,i]) + list(XX[:,i]) )
    X[:,i]  = lbl.transform(X[:,i])
    XX[:,i] = lbl.transform(XX[:,i])

X   = X.astype(float)
XX  = XX.astype(float)
y   = np.array(y).astype(float)

y_hat  = np.zeros((X.shape[0], experiment['n_models']))
XX_hat = np.zeros((XX.shape[0], experiment['n_models']))


print("\nafter array + LabelEncoder + asfloat")
print("Train: " + str(X.shape))
print("Test: " + str(XX.shape))
print("y: " + str(y.shape))
print("XX_hat: " + str(XX_hat.shape))
print("y_hat: " + str(y_hat.shape))

# set scaler
scaler  = StandardScaler()
scaler.fit(X)

# ------------------------------
#  Layer 1
# ------------------------------

folds   = list(StratifiedKFold(y, experiment['n_folds']))
scores  = np.zeros(experiment['n_models'])
score_weights = np.zeros(experiment['n_models'])   # r2 metric
gini_weights  = np.zeros(experiment['n_models'])

print("\n layer 1")
for k, model in enumerate(experiment['models']):
    # k = 0; model = experiment['models'][0]
    os.system("say model %s " % str(k))
    print("\n    model %s: \n%s " % (k+1,model) )
    # (n',K)
    # XX_hat_fold: (n', n_folds): memorizes the model prediction of XX
    XX_hat_fold = np.zeros((XX.shape[0], len(folds)))
    score_fold  = np.zeros( len(folds))
    gini_fold   = np.zeros( len(folds))

    if k in [2,3,4]:
        print('scaling XX')
        XX = scaler.transform(XX)

    for i, (train, test) in enumerate(folds):
        # i = 0; train = folds[0][0]; test = folds[0][1]
        os.system("say Fold %s " % str(i))
        print("\nFold", i)
        X_train = X[train]
        y_train = y[train]
        X_test  = X[test]
        y_test  = y[test]

        if k in [2,3,4]:
            print('scaling')
            X_train = scaler.transform(X_train)
            X_test  = scaler.transform(X_test)

        model.fit(X_train, y_train)
        y_hat[test, k] = model.predict(X_test)

        score_fold[i]  = r2_score(y_test, y_hat[test, k])
        gini_fold[i]   = gal_fct.gini_normalized(y_test, y_hat[test, k])
        print("CV r2:   %0.3f" % score_fold[i])
        print("CV gini: %0.3f" % gini_fold[i])

        XX_hat_fold[:, i] = model.predict(XX)

    # average of all models predictions for XX
    XX_hat[:,k]      = XX_hat_fold.mean(1)
    score_weights[k] = np.mean(score_fold)
    gini_weights[k]  = np.mean(gini_fold)

# ------------------------------
#  Layer 2
# ------------------------------
print('layer 2')
os.system("say Second Layer")

cv_scores = [ 0.10596634,  0.07655227, -0.02556798, -2.09025279, -1.62772704]

model  = LinearRegression()
cy_hat = y_hat / y_hat.mean(0)
model.fit(cy_hat, y)
# 1) standard mean
# predict
os.system("say Predict with standard mean")
yy_pred = model.predict(XX_hat)

preds = pd.DataFrame({"Id": raw_test_index, "Hazard": yy_pred})
preds = preds.set_index('Id')
preds.to_csv('../submissions/stack_mean_02.csv')

# 2) weighted by cv_scores
# this did not result in a better score
os.system("say Predict with cross validation weights")
XX_hat_weighted = XX_hat * cv_scores
yy_pred_w = model.predict(XX_hat_weighted)
preds = pd.DataFrame({"Id": raw_test_index, "Hazard": yy_pred_w})
preds = preds.set_index('Id')
preds.to_csv('../submissions/stack_cv_weights_03.csv')

# 3) weighted by r2 folds
# this also did not result in a better score
os.system("say Predict with fold weights")
XX_hat_w_folds = XX_hat * score_weights
yy_pred_w_f = model.predict(XX_hat_w_folds)
preds = pd.DataFrame({"Id": raw_test_index, "Hazard": yy_pred_w_f})
preds = preds.set_index('Id')
preds.to_csv('../submissions/stack_fold_weights_04.csv')

# 4) weighted by gini weights
# This weighting increased the final score by 0.01!
os.system("say Predict with fold weights")
XX_hat_gw = XX_hat * gini_weights
yy_pred_gw = model.predict(XX_hat_gw)
preds = pd.DataFrame({"Id": raw_test_index, "Hazard": yy_pred_gw})
preds = preds.set_index('Id')
preds.to_csv('../submissions/stack_gini_weights_05.csv')
