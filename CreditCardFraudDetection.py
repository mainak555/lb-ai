#%% [markdown]
## Credit Card Fraud Detection  
> Using Keras & AutoEncoders
#%%
import os
try:
    print(os.getcwd())
    os.chdir(os.path.join(os.getcwd(), '../aiml-for-all/kaggle/Credit-Card-Fraud-Detection'))
    print(os.getcwd())
except:
    print('Error in Setting Path')
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
df = pd.read_csv('creditcard.csv')
df[:5]
#%%
