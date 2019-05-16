#%% [markdown]
# Q2. Design a code to perform logistic regression on Iris dataset. 
## Make sure to use cross entropy to calculate loss values. 
## Don't use any prebuilt function for logistic regression. Write it on your own
#%%
from sklearn.datasets import load_iris
iris = load_iris()
print("\
Keys: {}\n\
Data Shape: {}\n\
Type: {} [{}D]\n\
Feature Names: {}\n\
Description: {}\
".format(iris.keys(), iris.data.shape, 
    type(iris.data), iris.data.ndim,
    iris.feature_names, iris.DESCR))
#%%
import numpy as np
import matplotlib.pyplot as plt
X = iris.data[:, :2]        #Take the first two features
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
#%%
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
fig = plt.figure()
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=iris['target'],
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
plt.show()
#%%
class LogisticRegression:
    def __init__(self, alpha = 0.0001, iteration = 10000, verbose = (True, 1000)):
        self.alpha = alpha
        self.iteration = iteration
        self.verbose = verbose
        self.theta = np.ndarray
        self.cost = np.ndarray
    
    def print_(self, text, skip=True):        
        if(self.verbose[0] and not skip):
            print(text)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def logistic_sigmoid(self, x):
        return np.exp(x) / 1 + np.exp(x)

    def hx(self, theta, X, n):        
        m = X.shape[0]                                              #Nos. of Records/Rows
        h = np.ones((m, 1))                                         #initializing with 1 for all rows
        theta = theta.reshape(1, n+1)                               #[[theta0....thetaN]] 1D->2D
        for i in range(0, m):
            h[i] = self.sigmoid(float(np.matmul(theta, X[i])))      #1/(1+e^-(theta.T*X))
            #h[i] = np.exp(np.matmul(theta, X[i]))                   #SoftMax Implementation
        #h = h / sum(h)                                              #SoftMax Implementation
        h = h.reshape(m)                                            #2D->1D/Flatten
        return h

    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()    #Cross Entropy

    def gradient_descent(self, theta, h, X, y, n):
        #w = np.random.randn(2)
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
            self.cost[i] = self.loss(h, y)
            self.print_('Cost/Iteration[{}]: {}'.format(i, self.cost[i]), not(iteration_count >= self.verbose[1]))
            #i +=1
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

    def predict(self, X, target_classes=(0, 1)):
        n = X.shape[1]
        m = X.shape[0]
        x0 = np.ones((m, 1))
        x = np.concatenate((x0, X), axis = 1)
        y_pred = self.hx(self.theta, x, n)        
        self.print_('Predicted y:\n{}'.format(y_pred))
        for i in range(0, m):
            if y_pred[i] > 0.5:
                y_pred[i] = target_classes[1]
            else:
                y_pred[i] = target_classes[0]
        return y_pred 

    def cost_minimization_curve(self, msg=''):
        plt.plot(np.arange(self.iteration), self.cost.tolist())
        plt.xlabel('cost')
        plt.ylabel('iteration')
        plt.title('Cost Minimisation. {}'.format(msg))
        plt.show()
#%%
def accuracy(y_hat, y_actual):
    k = 0
    for i in range(0, y_hat.shape[0]):
        if y_hat[i] == y_actual[i]:
            k += 1
    accuracy = k/y_actual.shape[0]
    return accuracy

def precision_recall(y_hat, y_actual, target_classes=(0, 1)):
    tp = fp = fn = 0
    precision = recall = np.inf
    for i in range(0, y_hat.shape[0]):
        if y_hat[i] == y_actual[i] == target_classes[0]:
            tp += 1
        elif y_hat[i] == target_classes[0] and y_actual[i] == target_classes[1]:
            fp += 1
        elif y_hat[i] == target_classes[1] and y_actual[i] == target_classes[0]:
            fn += 1
    try:
        precision = tp / (tp + fp)
    except:
        pass
    try:
        recall = tp / (tp + fn)  
    except:
        pass  
    return precision, recall
#%%
import sklearn.model_selection as skModel
X_train, X_test, y_train, y_test = skModel.train_test_split(iris['data'], iris['target'], random_state = 42)
print("X_train: ", X_train.shape, "X_test: ", X_test.shape)
#%%
import pandas as pd
def train_test_df(x_train, y_train, x_test, y_test):
    df_train = pd.DataFrame(X_train)
    df_train.columns = iris.feature_names
    df_train['class'] = y_train
    df_test = pd.DataFrame(X_test)
    df_test.columns = iris.feature_names
    df_test['class'] = y_test
    return df_train, df_test

df_train, df_test = train_test_df(X_train, y_train, X_test, y_test)
df_train[df_train['class'] != 2][:5]
#%%
def filter_by_column_value(df, colName, colValue):
    return df[df[colName] != colValue]

train_set_01 = filter_by_column_value(df_train, 'class', 2)
train_set_12 = filter_by_column_value(df_train, 'class', 0)
train_set_20 = filter_by_column_value(df_train, 'class', 1)

train_set_01[:5]
train_set_12[:5]
train_set_20[:5]
#%%
lm = LogisticRegression(iteration=20000, verbose=(True, 200))
theta_01 = lm.fit(train_set_01.iloc[:,:-1], train_set_01.iloc[:,-1:]['class'])
lm.cost_minimization_curve()
print('Thetas[01]: {}'.format(lm.theta))
y_pred_01 = lm.predict(train_set_01.iloc[:,:-1])
prec_rec = precision_recall(y_pred_01, train_set_01.iloc[:,-1:].reset_index()['class'])
print('Accuracy: {}, Precision: {}, Recall: {}'.format(
        accuracy(y_pred_01, train_set_01.iloc[:,-1:].reset_index()['class']),
        prec_rec[0], prec_rec[1]))
#%%
theta_12 = lm.fit(train_set_12.iloc[:,:-1], train_set_12.iloc[:,-1:]['class'])
print('Thetas[12]: {}'.format(lm.theta))
lm.cost_minimization_curve()
y_pred_12 = lm.predict(train_set_12.iloc[:,:-1])
prec_rec = precision_recall(y_pred_12, train_set_12.iloc[:,-1:].reset_index()['class'], (1, 0))
print('Accuracy: {}, Precision: {}, Recall: {}'.format(
        accuracy(y_pred_12, train_set_12.iloc[:,-1:].reset_index()['class']),
        prec_rec[0], prec_rec[1]))
#%%
theta_20 = lm.fit(train_set_20.iloc[:,:-1], train_set_20.iloc[:,-1:]['class'])
print('Thetas[20]: {}'.format(lm.theta))
lm.cost_minimization_curve()
y_pred_20 = lm.predict(train_set_20.iloc[:,:-1])
prec_rec = precision_recall(y_pred_20, train_set_20.iloc[:,-1:].reset_index()['class'], (2, 0))
print('Accuracy: {}, Precision: {}, Recall: {}'.format(
        accuracy(y_pred_20, train_set_20.iloc[:,-1:].reset_index()['class']),
        prec_rec[0], prec_rec[1]))
#%%

#%%
[]