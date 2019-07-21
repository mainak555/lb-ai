#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), './Advanced-DeepLearning-Keras'))
	print(os.getcwd())
except:
	pass

#%%
import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
#%%
dataPath = '/tmp/dataset/belgium_tsc'
#%%
if not os.path.isdir(dataPath):
    os.makedirs(dataPath, exist_ok=True)

#%%
#get_ipython().system('wget https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip -P /tmp/dataset/belgium_tsc')
#get_ipython().system('wget https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip -P /tmp/dataset/belgium_tsc')

#%%
#os.rename('BelgiumTSC_Testing.zip', data_dir + '/BelgiumTSC_Testing.zip')
#os.rename('BelgiumTSC_Training.zip', data_dir + '/BelgiumTSC_Training.zip')

#%%
# import zipfile
# with zipfile.ZipFile(dataPath + '/BelgiumTSC_Testing.zip', 'r') as zip_ref:
#     zip_ref.extractall(dataPath)
# with zipfile.ZipFile(dataPath + '/BelgiumTSC_Training.zip', 'r') as zip_ref:
#     zip_ref.extractall(dataPath)
#%%
# if os.path.exists(dataPath + '/BelgiumTSC_Testing.zip'):
#     os.remove(dataPath + '/BelgiumTSC_Testing.zip')
# if os.path.exists(dataPath + '/BelgiumTSC_Training.zip'):
#     os.remove(dataPath + '/BelgiumTSC_Training.zip')
#%%
def load_data(data_dir):
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir) 
                       if os.path.isdir(os.path.join(data_dir, d))]

    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                          for f in os.listdir(label_dir) 
                              if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels # these provide you data in numpy arrays
#%%
train_data_dir = os.path.join(dataPath, "Training")
train_images, train_labels = load_data(train_data_dir)
#%%
test_data_dir = os.path.join(dataPath, "Testing")
test_images, test_labels = load_data(test_data_dir)
#%%
def find_label_name(x):
    if x in [2,3,4,7,8,9,10,12,13,15,17,18,22,26,27,28,29,34,35]: 
        return "triangles"
    elif x in [36,43,48,50,55,56,57,58,59,61,65]: 
        return "redcircles"
    elif x in [72,75,76,78,79,80,81]: 
        return "bluecircles"
    elif x in [82,84,85,86]: 
        return "redbluecircles"
    elif x in [32,41]: 
        return "diamonds"
    elif x in [31]: 
        return "revtriangle"
    elif x in [39]: 
        return "stop",
    elif x in [42]: 
        return "forbidden"
    elif x in [118,151,155,181]: 
        return "squares"
    elif x in [37,87,90,94,95,96,97,149,150,163]:
        return "rectanglesup"
    elif x in [111,112]:
        return "rectanglesdown"
    else:
        return "undefined"
#%%
def display_images_and_labels(images64, labels, cmap='jet', channel=3):
    fig = plt.figure(figsize=(15,15))
    for i in range(20):
        index = np.random.randint(0, 1000)    
        plt.subplot(5, 4, i+1)
        plt.tight_layout()
        if channel == 3:
            plt.imshow(images64[index][:,:,:], cmap=cmap, interpolation=None)
        else:
            plt.imshow(images64[index][:,:,channel], cmap=cmap, interpolation=None)
        plt.title('Label: {} ({})'.format(labels[index], find_label_name(labels[index])))
        plt.xticks([])
        plt.yticks([])
    plt.show()
#%%
train_images[1][:,:,1]
# Resize images
train_images64 = [skimage.transform.resize(image, (64, 64), mode='constant')
                for image in train_images]
#%%
test_images64 = [skimage.transform.resize(image, (64, 64), mode='constant')
                for image in test_images]
#%%
display_images_and_labels(train_images64, train_labels)
print('Type: %s, Length: %s' %(type(train_images64), len(train_images64)))
#%%
display_images_and_labels(test_images64, test_labels)
print('Type: %s, Length: %s' %(type(test_images64), len(test_images64)))
#%%
def remove_undefined(images, labels):
    x = []
    y = []
    for i in labels:
        if find_label_name(i) != 'undefined':
            x.append(images[i])
            y.append(labels[i])
    return np.asarray(x), np.asarray(y)
#%%
x_train, y_train = remove_undefined(train_images64, train_labels)
print('Shape x: {}, y: {}'.format(x_train.shape, y_train.shape))
#%%
x_test, y_test = remove_undefined(test_images64, test_labels)
print('Shape x: {}, y: {}'.format(x_test.shape, y_test.shape))
#%%
x_train[:5]
#%%
display_images_and_labels(x_train, y_train)
display_images_and_labels(x_test, y_test)
#%%
np.unique(y_train)
np.unique(y_test)
#%%
y_train.reshape(-1,1)
#%%
# from keras.utils import to_categorical
# y_train_cat = to_categorical(y_train.reshape(-1,1), 3)
# y_train_cat
#%%
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
y_train_cat = enc.fit_transform(y_train.reshape(-1,1))
y_test_cat = enc.transform(y_test.reshape(-1,1))
#%%
enc.categories_
#%%
y_train_cat.toarray()
y_test_cat.toarray()
#%%
y_train_cat.toarray().shape
#enc.inverse_transform(y_train_cat)
#%%
'''Normalize data'''
pixcel_depth = 255
x_train = x_train.astype('float32') / pixcel_depth
x_test = x_test.astype('float32') / pixcel_depth

'''Improving Accuracy'''
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean
#%%
x_train.shape
x_test.shape
#%%
display_images_and_labels(x_train, y_train, channel=0)
display_images_and_labels(x_test, y_test, channel=0)
#%%
K.clear_session()
with K.tf.device('/device:XLA_GPU:0'):
    config = tf.ConfigProto(intra_op_parallelism_threads=4,
                            inter_op_parallelism_threads=4, 
                            allow_soft_placement=True,
                            log_device_placement = True,
                            device_count = {'CPU' : 1, 'GPU' : 2})
    session = tf.Session(config=config)
    K.set_session(session)
#%%
import importlib
resnet = importlib.import_module('ResNet')
#%%
depth = lambda ResNetVersion, N: N * 6 + 2 if ResNetVersion == 1 else N * 9 + 2
#%%
model = resnet.ResNet.V2(input_shape=x_train.shape[1:], depth=depth(2, 3), num_classes=3)
model.summary()
#%%
from keras.utils import plot_model
from keras.optimizers import Adam
metric = importlib.import_module('Metric')

model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(resnet.ResNet.lr_schedule(0)),
              metrics=['accuracy', metric.Metric.F1])
plot_model(model, to_file="Traffic_ResNetV2.png", show_shapes=True)
#%%
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'Traffic_ResNetV2_model.{epoch:03d}.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
#%%
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, TensorBoard

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(resnet.ResNet.lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

early_stopping = EarlyStopping(monitor='val_acc', patience=5)
tensor_board = TensorBoard(log_dir='/tmp/log_dir', histogram_freq=0, write_graph=True, write_images=True)

callbacks = [checkpoint, lr_reducer, lr_scheduler, early_stopping, tensor_board]
#%%
batch_size = 140
epoch = 20
#%%
model.fit(x_train, y_train_cat.toarray(),
          batch_size=batch_size,
          epochs=epoch,
          #validation_split=0.2,
          validation_data = (x_test, y_test_cat.toarray()),
          verbose=1,
          shuffle=True,
          callbacks=callbacks)
#%%
'''Test Validation'''
score = model.evaluate(x_test, y_test_cat.toarray(), verbose=1)
model.metrics_names
print('Loss: {}\
    \nAccuracy: {}\
    \nF1: {}'.format(score[0], score[1], score[2]))
#%%
'''Using real-time data augmentation'''
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(        
        featurewise_center=False,               # set input mean to 0 over the dataset        
        samplewise_center=False,                # set each sample mean to 0        
        featurewise_std_normalization=False,    # divide inputs by std of dataset        
        samplewise_std_normalization=False,     # divide each input by its std        
        zca_whitening=False,                    # apply ZCA whitening       
        rotation_range=0,                       # randomly rotate images in the range (deg 0 to 180)        
        width_shift_range=0.1,                  # randomly shift images horizontally        
        height_shift_range=0.1,                 # randomly shift images vertically        
        horizontal_flip=True,                   # randomly flip images        
        vertical_flip=False                     # randomly flip images
    )                
#%%
datagen.fit(x_train)
#%%
model.fit_generator(
        datagen.flow(x_train, y_train_cat.toarray(), batch_size=batch_size),
        validation_data=(x_test, y_test_cat.toarray()),
        epochs=epoch, verbose=1, workers=4,
        steps_per_epoch=len(x_train)//batch_size,
        callbacks=callbacks)
#%%
'''Test Validation'''
score = model.evaluate(x_test, y_test_cat.toarray(), verbose=1)
model.metrics_names
print('Loss: {}\
    \nAccuracy: {}\
    \nF1: {}'.format(score[0], score[1], score[2]))