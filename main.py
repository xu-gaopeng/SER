# coding:utf-8
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display
import os
import numpy.random as npr
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
import math
import pcc.pc
from sklearn import metrics
# data precession
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# model
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from  sklearn.model_selection  import  train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from math import sqrt
import json
x=np.load('D:/ser/Train/opensmile_features1582.npy')
x_test=np.load('D:/ser/Test/opensmile_features1582.npy')
X=x.astype(float)
x_test=x_test.astype(float)
y=np.loadtxt('D:/ser/Train/P_Train_27KB.txt')
y=np.array(y)
y=pd.DataFrame(y,columns=['id','P'])
s=y.P
id=y.drop('P',axis=1)
Y=np.array(s)
# y=np.loadtxt('D:/ser/Train/P_Train_27KB.txt')
# y=np.array(y)
# y=pd.DataFrame(y,columns=['id','P'])
# s=y.P
# Y=np.array(s)
# print X.shape
Y=Y.reshape(-1,1)

le=MinMaxScaler(feature_range=(-3,3))
le.fit(Y)
Y=le.transform(Y)
print Y.shape
train_x,valid_x,train_y,valid_y=train_test_split(X,Y,test_size=0.2,random_state=0)
plt.plot(Y,c='b')

# plt.title('P_Train')
# plt.legend(['Y'])
# le.fit(X)
# Y=le.transform(X)
# plt.plot(X,c='r')
#
# plt.title('X')
# plt.legend(['real'])
#Ensemble class

class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                print ("Fit Model %d fold %d" % (i, j))
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(axis=1)

        # results = cross_val_score(self.stacker, S_train, y, cv=5, scoring='r2')
        # print("Stacker score: %.4f (%.4f)" % (results.mean(), results.std()))
        # exit()

        self.stacker.fit(S_train, y)
        res = self.stacker.predict(S_test)[:]
        return res

# rf params
rf_params = {}
rf_params['n_estimators'] = 50
rf_params['max_depth'] = 8
rf_params['min_samples_split'] = 100
rf_params['min_samples_leaf'] = 30

# xgb params
xgb_params = {}
xgb_params['min_child_weight'] = 12
xgb_params['learning_rate'] = 0.37
xgb_params['max_depth'] = 6
xgb_params['subsample'] = 0.77
xgb_params['reg_lambda'] = 0.8
xgb_params['reg_alpha'] = 0.4
xgb_params['base_score'] = 0
# xgb_params['seed'] = 400
xgb_params['silent'] = 1

# lgb params
lgb_params = {}
lgb_params['max_bin'] = 8
lgb_params['min_bin'] = 1
lgb_params['min_data_in_bin'] = 1
lgb_params['learning_rate'] = 0.37
lgb_params['metric'] = 'l1'  #
lgb_params['sub_feature'] = 0.35
lgb_params['bagging_fraction'] = 0.85
lgb_params['bagging_freq'] = 40
lgb_params['num_leaves'] = 512
lgb_params['min_data'] = 500
lgb_params['min_hessian'] = 0.05
lgb_params['verbose'] = 0
lgb_params['feature_fraction_seed'] = 2
lgb_params['bagging_seed'] = 3

# XGB model
xgb_model = XGBRegressor(**xgb_params)

# lgb model
lgb_model = LGBMRegressor(**lgb_params)

# RF model
rf_model = RandomForestRegressor(**rf_params)

# ET model
et_model = ExtraTreesRegressor()

# SVR model
# SVM is too slow in more then 10000 set
svr_model = SVR(kernel='rbf', C=10.0, epsilon=0.01)

# DecsionTree model
dt_model = DecisionTreeRegressor()

# AdaBoost model
ada_model = AdaBoostRegressor()

stack = Ensemble(n_splits=5,
                 stacker=LinearRegression(),
                 base_models=(rf_model, xgb_model, et_model, ada_model))

y_valid = stack.fit_predict(train_x, train_y.ravel(), valid_x)

pre_vaild = y_valid
print "the result of 1582 dimensional features"
#metrics model
mse=metrics.mean_squared_error(pre_vaild,valid_y)
rmse=np.sqrt(mse)
r2=metrics.r2_score(pre_vaild,valid_y)
pcc=pc(pre_vaild,valid_y)
print 'pcc:{}'.format(pcc)
print 'mse:{}'.format(mse)
print 'rmse:{}'.format(rmse)
print 'r2:{}'.format(r2)

#predict y_pre
y_pre = stack.fit_predict(train_x, train_y.ravel(), x_test)
print y_pre
y_pre=np.array(y_pre)
y_pre=y_pre.ravel()
y_id=range(11401,12001)
y_id=np.array(y_id)
sub_dict = dict(zip(y_id, y_pre))


print "submit....."

with open("C:/Emotion-Recognition-from-Speech-master/src/submission_sample1.json","w") as f:


    json.dump(sub_dict,f,separators=('\n','\t'))
#plt predicted and real
# plt.plot(y_test,c='r')
# plt.plot(test_y,c='g')
# plt.title('predicted and real')
# plt.legend(['predicted','real'])