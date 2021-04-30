import numpy as np
import pandas as pd
import xgboost as xgb
import math
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
train_x,valid_x,train_y,valid_y=train_test_split(X,Y,test_size=0.2,random_state=0)
et_model = ExtraTreesRegressor()
et_model.fit(train_x,train_y)
pre_test=et_model.predict(valid_x)
print "the result of 1582 dimensional features"
#metrics model
mse=metrics.mean_squared_error(pre_test,valid_y)
rmse=np.sqrt(mse)
r2=metrics.r2_score(pre_test,valid_y)
import math


def calcMean(x,y):
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x+0.0)/n
    y_mean = float(sum_y+0.0)/n
    return x_mean,y_mean
def pc(x,y):
    x_mean,y_mean = calcMean(x,y)
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i]-x_mean)*(y[i]-y_mean)
    for i in range(n):
        x_pow += math.pow(x[i]-x_mean,2)
    for i in range(n):
        y_pow += math.pow(y[i]-y_mean,2)
    sumBottom = math.sqrt(x_pow*y_pow)
    p = sumTop/sumBottom
    return p

pcc=pc(pre_test,valid_y)
print 'pcc:{}'.format(pcc)
print 'mse:{}'.format(mse)
print 'rmse:{}'.format(rmse)
print 'r2:{}'.format(r2)



#plt predicted and real
# plt.plot(y_test,c='r')
# plt.plot(test_y,c='g')
# plt.title('predicted and real')
# plt.legend(['predicted','real'])

