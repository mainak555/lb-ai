
#%%
import numpy as np
class ActivationFunction:
    @staticmethod
    def sigmoid(x, derivative=False):
        if not derivative:
            return 1 / (1 + np.exp(-x))
        fx = ActivationFunction.sigmoid(x)
        return fx * (1 - fx)

    @staticmethod
    def softmax(x, derivative=False):
        if not derivative:
            return np.exp(x) / sum(np.exp(x))
    
    @staticmethod
    def tanh(x, derivative=False):
        if not derivative:
            return np.tanh(x)
        return 1 - ActivationFunction.tanh(x) ** 2
   
    @staticmethod
    def identity(x, derivative=False):
        if not derivative:
            return x
        return 1
    
    @staticmethod
    def arcTan(x, derivative=False):
        if not derivative:
            np.arctan(x)
        return 1 / (1 + x**2)
#%%
import matplotlib.pyplot as plt
#%%
#Sigmoid
fig, ax = plt.subplots(4, 2, sharex='all', sharey='all', figsize=(10, 20))
x = np.arange(-10, 10, 0.001)
for i in range(4):
    ax[i, 0].set_title('f(x)')
    fx = ActivationFunction.sigmoid(x)
    ax[i, 0].plot(x, fx)
    ax[i, 1].set_title('d/dx f(x)')
    ddx_of_fx = ActivationFunction.sigmoid(x, derivative=True)
    ax[i, 1].plot(x, ddx_of_fx)
    x = fx
plt.show()
#%%
#Tanh
fig, ax = plt.subplots(8, 2, sharex='all', sharey='all', figsize=(10, 20))
x = np.arange(-10, 10, 0.001)
for i in range(4):
    ax[i, 0].set_title('f(x)')
    fx = ActivationFunction.tanh(x)
    ax[i, 0].plot(x, fx)
    ax[i, 1].set_title('d/dx f(x)')
    ddx_of_fx = ActivationFunction.tanh(x, derivative=True)
    ax[i, 1].plot(x, ddx_of_fx)
    x = fx
plt.show()
#%%
#ArcTan
fig, ax = plt.subplots(8, 2, sharex='all', sharey='all', figsize=(10, 20))
x = np.arange(-10, 10, 0.001)
for i in range(4):
    ax[i, 0].set_title('f(x)')
    fx = ActivationFunction.arcTan(x)
    ax[i, 0].plot(x, fx)
    ax[i, 1].set_title('d/dx f(x)')
    ddx_of_fx = ActivationFunction.arcTan(x, derivative=True)
    ax[i, 1].plot(x, ddx_of_fx)
    x = fx
plt.show()
#%%
x = np.arange(-10, 10, 0.001)
plt.plot(x, ActivationFunction.softmax(x))
plt.show()