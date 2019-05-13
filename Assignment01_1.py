#%% [markdown]
# Q1. Design a code to run linear regression with Boston housing data present in sklearn.datasets. 
## Make a simple linear classifier and use gradient descent to optimize the loss. 
## Don't use any prebuilt function for linear regression. Write it on your own.
#%%
from sklearn.datasets import load_boston
boston = load_boston()
#%%
print("\
Keys: {}\n\
Data Shape: {}\n\
Type: {} [{}D]\n\
Feature Names: {}\n\
Description: {}\
".format(boston.keys(), boston.data.shape, 
    type(boston.data), boston.data.ndim,
    boston.feature_names, boston.DESCR))
#%%
import numpy as np
import pandas as pd
df = pd.DataFrame(boston.data)
df.columns = boston.feature_names
df[:10]
#%%
'''Adding Target 'Price' column to df'''
df['price'] = boston.target
df.describe()
#%%
import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
#%%
import matplotlib.pyplot as plt
df.hist(bins=20, figsize=(20,15))
plt.show()
#print(boston.DESCR)
'''
- CRIM     per capita crime rate by town
- ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS    proportion of non-retail business acres per town
- CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- NOX      nitric oxides concentration (parts per 10 million)
- RM       average number of rooms per dwelling
- AGE      proportion of owner-occupied units built prior to 1940
- DIS      weighted distances to five Boston employment centres
- RAD      index of accessibility to radial highways
- TAX      full-value property-tax rate per $10,000
- PTRATIO  pupil-teacher ratio by town
- B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT    % lower status of the population
- MEDV     Median value of owner-occupied homes in $1000's
'''
#%%
corr_matrix = df.corr()
corr_matrix[df.columns].style.background_gradient(cmap='coolwarm')

from pandas.plotting import scatter_matrix
scatter_matrix(df[df.columns], figsize=(15,15), alpha=0.2, diagonal='kde')
plt.show()

cnt = int (df.columns.size/2)
gr1 = df.columns[:cnt].tolist()
gr1.append('price')
#df[gr1][:10]
scatter_matrix(df[gr1], figsize=(15,15), alpha=0.2, diagonal='kde')
plt.show()
scatter_matrix(df[df.columns[7:]], figsize=(15,15), alpha=0.2, diagonal='kde')
plt.show()

scatter_matrix(df[['price', 'RM', 'LSTAT']], figsize=(12,8), diagonal='kde')
plt.show()
#%%
'''Train Test Split'''
from zlib import crc32
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda x: test_set_check(x, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

df_with_id = df.reset_index()
train, test = split_train_test_by_id(df_with_id, 0.2, 'index')
print('Train: {}, Test: {}'.format(len(train), len(test)))
train.head()
test.tail()
#%%
'''Functions'''
#f(x) = thita0*x0 + thita1*x1 + thita2*x2 + .... + thitaN*xN
def fx(theta, X, n):
    m = X.shape[0]                              #Nos. of Records/Rows
    fx = np.ones((m, 1))
    theta = theta.reshape(1, n+1)
    for i in range(0, m):
        fx[i] = float(np.matmul(theta, X[i]))
    fx = fx.reshape(X.shape[0])
    return fx

def gradientDescent(alpha, iteration, theta, h, X, y, n):
    cost = np.ones(iteration)
    m = X.shape[0]
    for i in range(0, iteration):
        theta[0] = theta[0] - (alpha/m) * sum(h - y)
        for j in range(1, n+1):
            theta[j] = theta[j] - (alpha/m) * sum((h - y) * X.T[j])
        h = fx(theta, X, n)
        cost[i] = (1/m) * 0.5 * sum(np.square(h - y))
    theta = theta.reshape(1, n+1)
    return theta, cost

def linearReg(X, y, alpha, iteration):
    n = X.shape[1]                              #Nos. of Features
    x0 = np.ones((X.shape[0], 1))
    X = np.concatenate((x0, X), axis = 1)
    theta = np.zeros(n+1)
    h = fx(theta, X, n)
    theta, cost = gradientDescent(alpha, iteration, theta, h, X, y, n)
    return theta, cost
#%%
linearReg(train[['RM', 'LSTAT']], train['price'], 0.0001, 1000)
