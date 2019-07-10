#%% [markdown]
## Credit Card Fraud Detection  
# Using Keras & AutoEncoders
#%%
#!kaggle datasets download -d mlg-ulb/creditcardfraud
filePath = '/tmp/dataset'
fileName = 'creditcard.csv'
import kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files('mlg-ulb/creditcardfraud', path=filePath, unzip=True)
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
SEED = 42
df = pd.read_csv(filePath + '/' + fileName)
df[:5]
df.isnull().values.any()
#%%
'''Splitting X & Y'''
Y = df['Class']
X = df.drop(['Class'], axis=1)
X[:3]
Y[Y==1].shape                           #Total Fraud Classes
df[df['Class']==1].Amount.describe()    #Fraud Amount
df[df['Class']==0].Amount.describe()    #Non_fraud
#%%
# Plotting Time of Fraud Happen & Fraud Amount
Y.value_counts(sort=True).plot(kind='bar', rot=0)
plt.title('Non-Fraud/Fraud Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(range(2), ['Non Fraud', 'Fraud'])
plt.show()
#%% Dont' Run (Will Take Time to Display)
'''
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of Transaction vs. Amount/Class')
ax1.scatter(df[df['Class']==0].Time, df[df['Class']==0].Amount)
ax1.set_title('Non-Fraud')
ax2.scatter(df[df['Class']==1].Time, df[df['Class']==1].Amount)
ax2.set_title('Fraud')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.show()
'''
#%% Visualising Fraud Vs 2000 Non Fraud Data
'''visualize the nature of fraud and non-fraud transactions using T-SNE. 
T-SNE (t-Distributed Stochastic Neighbor Embedding) is a dataset decomposition technique 
which reduced the dimentions of data and produces only top n components with maximum information
'''
from sklearn.manifold import TSNE
non_fraud = df[df['Class'] == 0].sample(5000)
fraud = df[df['Class'] == 1]
df_viz = non_fraud.append(fraud).sample(frac=1).reset_index(drop=True)
x_viz = df_viz.drop(['Class'], axis=1).values
y_viz = df_viz['Class'].values

def tnse_plot(x, y):
    tsne = TSNE(n_components=2, random_state=0)
    x_t = tsne.fit_transform(x)
    #print(x_t)    
    plt.figure(figsize=(12, 8))
    plt.scatter(x_t[np.where(y == 0), 0], x_t[np.where(y == 0), 1], 
        marker='o', color='g', linewidths='1', alpha=0.8, label='NonFraud')
    plt.scatter(x_t[np.where(y == 1), 0], x_t[np.where(y == 1), 1], 
        marker='o', color='r', linewidths='1', alpha=0.8, label='Fraud')
    
    plt.legend(loc='best')
    plt.show()

tnse_plot(x_viz, y_viz)
#%% 
'''Spliting Train/Test'''
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=SEED)

'''Deleting Time Column as doing Feature Scaling & Reseting Row Indexes'''
train_x = train_x.drop(['Time'], axis=1).reset_index(drop=True)
train_y = train_y.reset_index(drop=True)

test_x = test_x.drop(['Time'], axis=1).reset_index(drop=True)
test_y = test_y.reset_index(drop=True)
#%%
'''Network: Auto-Encoder'''
from keras import Sequential, optimizers
from keras.layers import Dense
from keras.models import Model
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

np.random.seed(SEED)
def auto_encoder(input_dim):
    autoencoder = Sequential([
        #Encoder
        Dense(units=200, activation='tanh', 
            kernel_initializer = 'glorot_uniform',        
            activity_regularizer=regularizers.l2(0.0001),
            input_shape = (input_dim,)),
        Dense(units=50, activation='relu'),
        #Decoder
        Dense(units=50, activation='tanh', input_shape=(50,)),
        Dense(units=200, activation='tanh'),
        Dense(units=input_dim, activation='relu')
    ])
    autoencoder.summary()    
    return autoencoder

def encoder_form_auto_encoder(model):
    encoder = Sequential([
        model.layers[0],
        model.layers[1]
    ])
    return encoder
#%%
train_fraud, train_non_fraud = train_x[train_y == 1], train_x[train_y == 0]
train_fraud = preprocessing.MinMaxScaler().fit_transform(train_fraud)           #Can't do Feature Scalling Globally otherwise above line will break
train_non_fraud = preprocessing.MinMaxScaler().fit_transform(train_non_fraud)   #Can't do fraud/Non-Fraud segrigation
'''For Self Ref. Before Cleanup
#train_non_fraud = train_non_fraud.drop(['Time'], axis=1).reset_index(drop=True)
#train_non_fraud = preprocessing.MinMaxScaler().fit_transform(train_non_fraud.iloc[:, 1:])   #.iloc[:, 1:]: Excluding Index Column as added after reset_index(drop=False)
'''
#%% 
'''Summarize history for accuracy'''
def plot_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('accuracy & loss')
    plt.ylabel('accuracy & loss')
    plt.xlabel('epoch')
    plt.show()
#%% 
'''Training Auto-Encoder On Non-Fraud Partial Train data For Vizualization'''
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15),
    ModelCheckpoint(filepath='/tmp/autoenc.h5', monitor='val_loss', save_best_only=True),
    TensorBoard(log_dir='/tmp/log_dir', histogram_freq=0, write_graph=True, write_images=True)
]

model = auto_encoder(train_non_fraud.shape[1])
model.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
history = model.fit(train_non_fraud[:150000], train_non_fraud[:150000], 
    batch_size=1500, epochs=100, callbacks=callbacks, 
    shuffle=False, validation_split=0.25, verbose=1)
print(history.history.keys())
plot_history(history)
#%% 
'''Encodeding Partial Train Data for Visualizing'''
encoder = encoder_form_auto_encoder(model)
encoded_fraud = encoder.predict(train_fraud)
encoded_non_fraud = encoder.predict(train_non_fraud[148000:152000])

x_viz = np.append(encoded_non_fraud, encoded_fraud, axis=0)
y_viz = np.append(
    np.zeros(encoded_non_fraud.shape[0]),
    np.ones(encoded_fraud.shape[0]) 
)
tnse_plot(x_viz, y_viz)
#%% 
'''Encodeding Test Data Just for Visualizing before Predict'''
# Scaling Test data 
test_fraud, test_non_fraud = test_x[test_y == 1], test_x[test_y == 0]
test_fraud = preprocessing.MinMaxScaler().fit_transform(test_fraud)         #Can't do Feature Scalling Globally otherwise above line will break
test_non_fraud = preprocessing.MinMaxScaler().fit_transform(test_non_fraud)   

encoded_fraud = encoder.predict(test_fraud)
encoded_non_fraud = encoder.predict(test_non_fraud[5000:15000])
x_viz = np.append(encoded_non_fraud, encoded_fraud, axis=0)
y_viz = np.append(
    np.zeros(encoded_non_fraud.shape[0]),
    np.ones(encoded_fraud.shape[0]) 
)
tnse_plot(x_viz[3000:8000], y_viz[3000:8000])
#%% Network Not Used
def network(input_dim):
    nn = Sequential([
        Dense(units=64, activation='tanh',
            kernel_initializer = 'glorot_uniform',        
            activity_regularizer=regularizers.l2(0.001), 
            input_shape = (input_dim,)),
        Dense(units=64, activation='tanh'),
        Dense(units=1, activation='softmax'),
    ])
    return nn
#%%
'''Solutioning'''
#1. Training Auto-Encoder On Complete Non-Fraud Train Data
model = auto_encoder(train_non_fraud.shape[1])
model.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
history = model.fit(train_non_fraud, train_non_fraud, 
    batch_size=1500, epochs=150, callbacks=callbacks, 
    shuffle=False, validation_split=0.25, verbose=2)
plot_history(history)
#%%
#2. Encoding Complete Train Data (Fraud + Non-Fraud)
encoder = encoder_form_auto_encoder(model)
norm_train_x = preprocessing.MinMaxScaler().fit_transform(train_x)
train_encoded = encoder.predict(norm_train_x)
tnse_plot(train_encoded[5000:10000], train_y[5000:10000])
#%%
#3. Test Data AutoEncoder o/p Visualization
norm_test_x = preprocessing.MinMaxScaler().fit_transform(test_x)
test_autoencoder = model.predict(norm_test_x)
tnse_plot(test_autoencoder[25000:30000], test_y[25000:30000])
#%%
#4. Test Data AutoEncoder o/p Prediction Error Distribution
mse = np.mean((norm_test_x - test_autoencoder) ** 2, axis=1)
err_df = pd.DataFrame({
    'reconstruction_error': mse,
    'true_class': test_y
}).describe()
#%%
#5. Encode Test Data
test_encoder = encoder.predict(norm_test_x)
tnse_plot(test_encoder[25000:30000], test_y[25000:30000])
#%%
#6. Train SVM to Draw Boundary/Classify
from sklearn import svm, metrics
clf = svm.SVC(kernel='linear')
clf.fit(train_encoded, train_y)
predict = clf.predict(train_encoded)
print('Train Accuracy: ', metrics.accuracy_score(predict, train_y))
#%%
#7. Test Accuracy
predict = clf.predict(test_encoder)
print('Test Accuracy: ', metrics.accuracy_score(predict, test_y))
print('Test F1: ', metrics.f1_score(predict, test_y))
print('Confusion Matrix:\n', metrics.confusion_matrix(test_y, predict))
#%%

