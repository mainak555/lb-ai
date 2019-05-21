#%%
import numpy as np
class Squash:
    def __init__(self):
        self.function = {
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'swish': lambda x: x / (1 + np.exp(-x)),
            'tanh': lambda x: np.tanh(x),
            'softmax': lambda x, exps = np.exp(x): exps / float(sum(exps)),
            'arctan': lambda x: np.arctan(x),
            'identity': lambda x: x,

            }            
        
        self.derivative = {
            'sigmoid': lambda x, fx = self.function['sigmoid'](x): fx * (1 - fx),
            'swish': lambda x, fx = x * self.function['sigmoid'](x): fx + (1 - fx) * self.function['sigmoid'](x),
            'tanh': lambda x, fx = self.function['tanh'](x): 1 - fx**2,
            'softmax': lambda x, fx = self.function['softmax'](x): fx * (1 - fx),
            'arctan': lambda x: 1 / (1 + x**2),
            'identity': lambda x: 1,
        }
#%%
import matplotlib.pyplot as plt
activation = Squash()
z = np.arange(-10, 10, 0.001)
#%%
#Sigmoid
fig, ax = plt.subplots(4, 3, sharex='all', figsize=(10, 20))
x = z
for i in range(4):
    fx = activation.function['sigmoid'](x)
    ax[i, 0].set_title('f(x)')
    ax[i, 0].set_xlabel('x->')
    ax[i, 0].plot(x, fx)    
    ddx_of_fx = activation.derivative['sigmoid'](x)
    ax[i, 1].set_title('d/dx f(x)')
    ax[i, 1].set_xlabel('f(x)->')
    ax[i, 1].plot(fx, ddx_of_fx)
    ax[i, 2].set_title('d/dx f(x)')
    ax[i, 2].set_xlabel('input->')
    ax[i, 2].plot(z, ddx_of_fx)
    print('>{}\n-f(x): {}\n-d/dx f(x): {}'.format(i, fx, ddx_of_fx))
    x = fx
plt.show()
#%%
#Tanh
fig, ax = plt.subplots(4, 3, sharex='all', figsize=(10, 20))
x = z
for i in range(4):
    fx = activation.function['tanh'](x)
    ax[i, 0].set_title('f(x)')
    ax[i, 0].set_xlabel('x->')
    ax[i, 0].plot(x, fx)    
    ddx_of_fx = activation.derivative['tanh'](x)
    ax[i, 1].set_title('d/dx f(x)')
    ax[i, 1].set_xlabel('f(x)->')
    ax[i, 1].plot(fx, ddx_of_fx)
    ax[i, 2].set_title('d/dx f(x)')
    ax[i, 2].set_xlabel('input->')
    ax[i, 2].plot(z, ddx_of_fx)
    print('>{}\n-f(x): {}\n-d/dx f(x): {}'.format(i, fx, ddx_of_fx))
    x = fx
plt.show()
#%%
#ArcTan
fig, ax = plt.subplots(4, 3, sharex='all', figsize=(10, 20))
x = z
for i in range(4):
    fx = activation.function['arctan'](x)
    ax[i, 0].set_title('f(x)')
    ax[i, 0].set_xlabel('x->')
    ax[i, 0].plot(x, fx)    
    ddx_of_fx = activation.derivative['arctan'](x)
    ax[i, 1].set_title('d/dx f(x)')
    ax[i, 1].set_xlabel('f(x)->')
    ax[i, 1].plot(fx, ddx_of_fx)
    ax[i, 2].set_title('d/dx f(x)')
    ax[i, 2].set_xlabel('input->')
    ax[i, 2].plot(z, ddx_of_fx)
    print('>{}\n-f(x): {}\n-d/dx f(x): {}'.format(i, fx, ddx_of_fx))
    x = fx
plt.show()
#%%
#Softmax
fig, ax = plt.subplots(4, 3, sharex='all', figsize=(10, 20))
x = z
for i in range(4):
    fx = activation.function['softmax'](x)
    ax[i, 0].set_title('f(x)')
    ax[i, 0].set_xlabel('x->')
    ax[i, 0].plot(x, fx)    
    ddx_of_fx = activation.derivative['softmax'](x)
    ax[i, 1].set_title('d/dx f(x)')
    ax[i, 1].set_xlabel('f(x)->')
    ax[i, 1].plot(fx, ddx_of_fx)
    ax[i, 2].set_title('d/dx f(x)')
    ax[i, 2].set_xlabel('input->')
    ax[i, 2].plot(z, ddx_of_fx)
    print('>{}\n-f(x): {}\n-d/dx f(x): {}'.format(i, fx, ddx_of_fx))
    x = fx
plt.show()
#%%
#Swish
fig, ax = plt.subplots(4, 3, sharex='all', figsize=(10, 20))
x = z
for i in range(4):
    fx = activation.function['swish'](x)
    ax[i, 0].set_title('f(x)')
    ax[i, 0].set_xlabel('x->')
    ax[i, 0].plot(x, fx)    
    ddx_of_fx = activation.derivative['swish'](x)
    ax[i, 1].set_title('d/dx f(x)')
    ax[i, 1].set_xlabel('f(x)->')
    ax[i, 1].plot(fx, ddx_of_fx)
    ax[i, 2].set_title('d/dx f(x)')
    ax[i, 2].set_xlabel('input->')
    ax[i, 2].plot(z, ddx_of_fx)
    print('>{}\n-f(x): {}\n-d/dx f(x): {}'.format(i, fx, ddx_of_fx))
    x = fx
plt.show()