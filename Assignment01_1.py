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

# cnt = int (df.columns.size/2)
# gr1 = df.columns[:cnt].tolist()
# gr1.append('price')
# #df[gr1][:10]
# scatter_matrix(df[gr1], figsize=(15,15), alpha=0.2, diagonal='kde')
# plt.show()
# scatter_matrix(df[df.columns[7:]], figsize=(15,15), alpha=0.2, diagonal='kde')
# plt.show()

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
'''Functions
f(x) = thita0*x0 + thita1*x1 + thita2*x2 + .... + thitaN*xN
     = [thita0, theta1,...thetaN]*[x0 = 1
                                   x1
                                   .
                                   .
                                   xN]
'''
class LinearRegression:    
    def __init__(self, alpha = 0.0001, iteration = 10000, verbose = (True, 1000)):
        self.alpha = alpha
        self.iteration = iteration
        self.verbose = verbose
        self.theta = np.ndarray
        self.cost = np.ndarray
        #self.cost = np.empty((0, 0), int)

    def print_(self, text, skip=True):        
        if(self.verbose[0] and not skip):
            print(text)

    def hx(self, theta, X, n):
        m = X.shape[0]                              #Nos. of Records/Rows
        fx = np.ones((m, 1))                        #initializing with 1 for all rows
        theta = theta.reshape(1, n+1)               #[[theta0....thetaN]] 1D->2D
        for i in range(0, m):
            fx[i] = float(np.matmul(theta, X[i]))   #theta.T * X        
        fx = fx.reshape(m)                          #2D->1D
        return fx
    
    def gradient_descent(self, theta, h, X, y, n):
        w = np.random.randn(2)
        iteration_count = 0
        m = X.shape[0]
        self.cost = np.ones(self.iteration)
        for i in range(0, self.iteration):
        #i=0
        #while True:
            if(iteration_count >= self.verbose[1]):
                iteration_count = 0 
            else:
                iteration_count += 1
                
            theta[0] = theta[0] - (self.alpha/m) * sum(h - y)   #* x0 Omitted as = 1
            for j in range(1, n+1):
                theta[j] = theta[j] - (self.alpha/m) * sum((h - y) * X.T[j])
            h = self.hx(theta, X, n)
            self.cost[i] = (1/m) * 0.5 * sum(np.square(h - y))
            self.print_('Cost/Iteration[{}]: {}'.format(i, self.cost[i]), not(iteration_count >= self.verbose[1]))
            i +=1
        self.theta = theta.reshape(1, n+1)
        
    def fit(self, X, y, algo='gradient_descent'):
        n = X.shape[1]                              #Nos. of Features [x1....xN]
        m = X.shape[0]                              #no. of Rows/Records
        x0 = np.ones((m, 1))
        x = np.concatenate((x0, X), axis = 1)       #New Vector X: [x0, x1...xN]
        theta = np.zeros(n+1)                       #Initialize theta: [theta0...thetaN]
        fx = self.hx(theta, x, n)                   #Initial h for all rows = 1
        self.print_('Initial h(x):\n{}'.format(fx))
        self.gradient_descent(theta, fx, x, y, n)        
        return self.theta

    def predict(self, X):
        n = X.shape[1]
        m = X.shape[0]
        x0 = np.ones((m, 1))
        x = np.concatenate((x0, X), axis = 1)
        y_pred = self.hx(self.theta, x, n)
        self.print_('Predicted y:\n{}'.format(y_pred))
        return y_pred  
      
#%%
def RMSE(y_pred, y_actual):
    squared_err = (y_pred - y_actual) ** 2
    mse = np.mean(squared_err)
    return np.sqrt(mse)

def featureScaling(X_train):
    n = X_train.shape[1]
    m = X_train.shape[0]
    mean = np.ones(n)
    std = np.ones(n)
    for i in range(0, n-1):
        mean[i] = np.mean(X_train.T[i])
        std[i] = np.std(X_train.T[i])
        for j in range(0, m-1):
            X_train.iloc[j][i] = (X_train.iloc[j][i] - mean[i]) / std[i]
    return X_train
#%%
trained_scaled = featureScaling(train[['RM', 'LSTAT']])
lm = LinearRegression(iteration=10, verbose=(True, 200))
#lm.fit(train[['RM', 'LSTAT']], train['price'])
#print('RM: \n{}\LS:\n{}'.format(trained_scaled['RM'], trained_scaled['LSTAT']))
lm.fit(trained_scaled, train['price'])
print('Thetas: {}'.format(lm.theta))
#%%
iterations = np.arange(lm.iteration)
plt.plot(iterations, lm.cost.tolist())
plt.show()
#%%
y_pred = lm.predict(test[['RM', 'LSTAT']])
print('RMSE: {}'.format(RMSE(y_pred, test['price'])))
#%%
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
axis_x_train = list(train['RM'].T)
axis_y_train = list(train['LSTAT'].T)
axis_z_train = list(train['price'])

fig = mpl.pyplot.figure()
#fig = mpl.pyplot.figure(figsize=(8, 20))

ax = Axes3D(fig)
#ax = fig.add_subplot(2, 2, 1, projection='3d')

ax.scatter(axis_x_train, axis_y_train, axis_z_train, c=axis_z_train, cmap='Greens')
#ax.set_xlabel('RM')
#ax.set_ylabel('LSTAT')
#ax.set_zlabel('price')
#ax.set_title('Train Set')

axis_x_test = list(test['RM'].T)
axis_y_test = list(test['LSTAT'].T)
axis_z_test = list(test['price'])

ax.scatter(axis_x_test, axis_y_test, axis_z_test, c=axis_z_test, cmap='Reds')
ax.set_xlabel('RM')
ax.set_ylabel('LSTAT')
ax.set_zlabel('price')
ax.set_title('Train & Test Set')
#plt.show()

ax.scatter(axis_x_test, axis_y_test, y_pred, c=y_pred, cmap='Blues')
plt.show()