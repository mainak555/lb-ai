#%%
import numpy as np
class Squash:
    def __init__(self):
        self.function = {
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'tanh': lambda x: np.tanh(x),
            'arctan': lambda x: np.arctan(x),
            'softmax': lambda x: (lambda exps= np.exp(x): exps / float(sum(exps)))(),
            'softplus': lambda x: np.log(1 + np.exp(x)),
            'swish': lambda x: x / (1 + np.exp(-x)),
            'identity': lambda x: x,
            'relu': lambda x: [item if item >= 0 else 0 for item in x],
            'prelu': lambda x, alpha: [item if item >= 0 else alpha * item for item in x],
            'elu': lambda x, alpha: [item if item >= 0 else np.dot(alpha, np.exp(item) - 1) for item in x], #Leaky Relu
            'arctanh': lambda x: np.arctanh(x) 
            }            
        
        self.derivative = {
            'sigmoid': lambda x: (lambda fx= self.function['sigmoid'](x): fx * (1 - fx))(),
            'tanh': lambda x: (lambda fx = self.function['tanh'](x): 1 - fx**2)(),
            'arctan': lambda x: 1 / (1 + x**2),
            'softmax': lambda x: (lambda fx= self.function['softmax'](x): fx * (1 - fx))(),
            'softplus': lambda x: self.function['sigmoid'](x),
            'swish': lambda x: (lambda x= x, fx= x * self.function['sigmoid'](x): fx + (1 - fx) * self.function['sigmoid'](x))(),
            'identity': lambda x: np.ones(x.shape[0]),
            'relu': lambda x: [1 if item >= 0 else 0 for item in x],
            'prelu': lambda x, alpha: [1 if item >= 0 else alpha for item in x],
            'elu': lambda x, alpha: [1 if item >= 0 else sum(self.function['elu']([item], alpha), alpha) for item in x],
            'arctanh': lambda x: 1 / (1- x**2)
        }
#%%
import matplotlib.pyplot as plt
activation = Squash()
z = np.arange(-5, 5, 0.1)
prn = lambda i, fx, d_fx: print('>>{}\n-f(x):\n{}\n-df/dx:\n{}'.format(i, fx, d_fx))
#%%
#Sigmoid
fig, ax = plt.subplots(4, 3, figsize=(10, 20))
a = z
for i in range(4):
    fx = activation.function['sigmoid'](a)
    ax[i, 0].set_title('f(x)')
    ax[i, 0].set_xlabel('x->')
    ax[i, 0].plot(a, fx)    
    ddx_of_fx = activation.derivative['sigmoid'](a)
    ax[i, 1].set_title('df/dx')
    ax[i, 1].set_xlabel('f(x)->')
    ax[i, 1].plot(fx, ddx_of_fx)
    ax[i, 2].set_title('df/dx vs. inp')
    ax[i, 2].set_xlabel('inp->')
    ax[i, 2].plot(z, ddx_of_fx)
    prn(i, fx, ddx_of_fx)
    a = fx
plt.show()
#%%
#Tanh
fig, ax = plt.subplots(4, 3, figsize=(10, 20))
a = z
for i in range(4):
    fx = activation.function['tanh'](a)
    ax[i, 0].set_title('f(x)')
    ax[i, 0].set_xlabel('x->')
    ax[i, 0].plot(a, fx)    
    ddx_of_fx = activation.derivative['tanh'](a)
    ax[i, 1].set_title('df/dx')
    ax[i, 1].set_xlabel('f(x)->')
    ax[i, 1].plot(fx, ddx_of_fx)
    ax[i, 2].set_title('df/dx vs. inp')
    ax[i, 2].set_xlabel('inp->')
    ax[i, 2].plot(z, ddx_of_fx)
    prn(i, fx, ddx_of_fx)
    a = fx
plt.show()
#%%
#ArcTan
fig, ax = plt.subplots(4, 3, figsize=(10, 20))
a = z
for i in range(4):
    fx = activation.function['arctan'](a)
    ax[i, 0].set_title('f(x)')
    ax[i, 0].set_xlabel('x->')
    ax[i, 0].plot(a, fx)    
    ddx_of_fx = activation.derivative['arctan'](a)
    ax[i, 1].set_title('df/dx')
    ax[i, 1].set_xlabel('f(x)->')
    ax[i, 1].plot(fx, ddx_of_fx)
    ax[i, 2].set_title('df/dx vs. inp')
    ax[i, 2].set_xlabel('inp->')
    ax[i, 2].plot(z, ddx_of_fx)
    prn(i, fx, ddx_of_fx)
    a = fx
plt.show()
#%%
#Softmax
fig, ax = plt.subplots(4, 3, figsize=(10, 20))
a = z
for i in range(4):
    fx = activation.function['softmax'](a)
    ax[i, 0].set_title('f(x)')
    ax[i, 0].set_xlabel('x->')
    ax[i, 0].plot(a, fx)    
    ddx_of_fx = activation.derivative['softmax'](a)
    ax[i, 1].set_title('df/dx')
    ax[i, 1].set_xlabel('f(x)->')
    ax[i, 1].plot(fx, ddx_of_fx)
    ax[i, 2].set_title('df/dx vs. inp')
    ax[i, 2].set_xlabel('inp->')
    ax[i, 2].plot(z, ddx_of_fx)
    prn(i, fx, ddx_of_fx)
    a = fx
plt.show()
#%%
#SoftPlus
fig, ax = plt.subplots(4, 3, figsize=(10, 20))
a = z
for i in range(4):
    fx = activation.function['softplus'](a)
    ax[i, 0].set_title('f(x)')
    ax[i, 0].set_xlabel('x->')
    ax[i, 0].plot(a, fx)    
    ddx_of_fx = activation.derivative['softplus'](a)
    ax[i, 1].set_title('df/dx')
    ax[i, 1].set_xlabel('f(x)->')
    ax[i, 1].plot(fx, ddx_of_fx)
    ax[i, 2].set_title('df/dx vs. inp')
    ax[i, 2].set_xlabel('inp->')
    ax[i, 2].plot(z, ddx_of_fx) 
    prn(i, fx, ddx_of_fx)   
    a = fx
plt.show()
#%%
#Swish
fig, ax = plt.subplots(4, 3, figsize=(10, 20))
a = z
for i in range(4):
    fx = activation.function['swish'](a)
    ax[i, 0].set_title('f(x)')
    ax[i, 0].set_xlabel('x->')
    ax[i, 0].plot(a, fx)    
    ddx_of_fx = activation.derivative['swish'](a)
    ax[i, 1].set_title('df/dx')
    ax[i, 1].set_xlabel('f(x)->')
    ax[i, 1].plot(fx, ddx_of_fx)
    ax[i, 2].set_title('df/dx vs. inp')
    ax[i, 2].set_xlabel('inp->')
    ax[i, 2].plot(z, ddx_of_fx) 
    prn(i, fx, ddx_of_fx)   
    a = fx
plt.show()
#%%
#Identity
fig, ax = plt.subplots(2, 3, figsize=(10, 10))
a = z
for i in range(2):
    fx = activation.function['identity'](a)
    ax[i, 0].set_title('f(x)')
    ax[i, 0].set_xlabel('x->')
    ax[i, 0].plot(a, fx)    
    ddx_of_fx = activation.derivative['identity'](a)
    ax[i, 1].set_title('df/dx')
    ax[i, 1].set_xlabel('f(x)->')
    ax[i, 1].plot(fx, ddx_of_fx)
    ax[i, 2].set_title('df/dx vs. inp')
    ax[i, 2].set_xlabel('inp->')
    ax[i, 2].plot(z, ddx_of_fx) 
    prn(i, fx, ddx_of_fx)   
    a = fx
plt.show()
#%%
#ReLu
fig, ax = plt.subplots(4, 3, figsize=(10, 20))
a = z
for i in range(4):
    fx = activation.function['relu'](a)
    ax[i, 0].set_title('f(x)')
    ax[i, 0].set_xlabel('x->')
    ax[i, 0].plot(a, fx)    
    ddx_of_fx = activation.derivative['relu'](a)
    ax[i, 1].set_title('df/dx')
    ax[i, 1].set_xlabel('f(x)->')
    ax[i, 1].plot(fx, ddx_of_fx)
    ax[i, 2].set_title('df/dx vs. inp')
    ax[i, 2].set_xlabel('inp->')
    ax[i, 2].plot(z, ddx_of_fx) 
    prn(i, fx, ddx_of_fx)   
    a = fx
plt.show()
#%%
#PReLu
fig, ax = plt.subplots(4, 3, figsize=(10, 20))
a = z
alpha = 2.25
for i in range(4):
    fx = activation.function['prelu'](a, alpha)
    ax[i, 0].set_title('f(x)')
    ax[i, 0].set_xlabel('x->')
    ax[i, 0].plot(a, fx)    
    ddx_of_fx = activation.derivative['prelu'](a, alpha)
    ax[i, 1].set_title('df/dx')
    ax[i, 1].set_xlabel('f(x)->')
    ax[i, 1].plot(fx, ddx_of_fx)
    ax[i, 2].set_title('df/dx vs. inp')
    ax[i, 2].set_xlabel('inp->')
    ax[i, 2].plot(z, ddx_of_fx) 
    prn(i, fx, ddx_of_fx)   
    a = fx
plt.show()
#%%
#ELu
fig, ax = plt.subplots(2, 3, figsize=(10, 10))
a = z
for i in range(2):
    fx = activation.function['elu'](a, alpha)
    ax[i, 0].set_title('f(x)')
    ax[i, 0].set_xlabel('x->')
    ax[i, 0].plot(a, fx)    
    ddx_of_fx = activation.derivative['elu'](a, alpha)
    ax[i, 1].set_title('df/dx')
    ax[i, 1].set_xlabel('f(x)->')
    ax[i, 1].plot(fx, ddx_of_fx)
    ax[i, 2].set_title('df/dx vs. inp')
    ax[i, 2].set_xlabel('inp->')
    ax[i, 2].plot(z, ddx_of_fx) 
    prn(i, fx, ddx_of_fx)   
    a = fx
plt.show()
#%%
#ArcTanh
fig, ax = plt.subplots(2, 3, figsize=(10, 10))
z = np.arange(-1, 1, .01)
a = z
for i in range(2):
    fx = activation.function['arctanh'](a)
    ax[i, 0].set_title('f(x)')
    ax[i, 0].set_xlabel('x->')
    ax[i, 0].plot(a, fx)    
    ddx_of_fx = activation.derivative['arctanh'](a)
    ax[i, 1].set_title('df/dx')
    ax[i, 1].set_xlabel('f(x)->')
    ax[i, 1].plot(fx, ddx_of_fx)
    ax[i, 2].set_title('df/dx vs. inp')
    ax[i, 2].set_xlabel('inp->')
    ax[i, 2].plot(z, ddx_of_fx) 
    prn(i, fx, ddx_of_fx)   
    a = fx
plt.show()