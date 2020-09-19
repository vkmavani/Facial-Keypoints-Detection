#!/usr/bin/env python
# coding: utf-8

# ## **Importing libraries**

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import MaxPooling2D, GlobalAvgPool2D
from tensorflow.keras.layers import Add, ReLU, Dense, Dropout, Flatten
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model


# ## **Load and Process data**
 
key_points = pd.read_csv('data\\training_frames_keypoints.csv')

def process_data(key_points):
    
    img_names = key_points['Unnamed: 0'].values
    X = []
    Y = []
    n = 0

    for file in img_names:
        img = cv2.imread("data\\training\\" + file)
        x,y,z = img.shape
        img = cv2.resize(img,(224,224))

        key_pts = key_points.iloc[n, 1:].to_numpy()
        key_pts = key_pts.astype('float').reshape(-1, 2)
        key_pts = key_pts * [224 / y, 224 / x]

        image_copy = np.copy(img)
        key_pts_copy = np.copy(key_pts)

        image_copy = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        image_copy=  image_copy/255.0

        key_pts_copy = (key_pts_copy - 100)/50.0

        X.append(image_copy)
        Y.append(key_pts_copy)

        n += 1

    return X,Y


X, y = process_data(key_points)


X = np.array(X)
y = np.array(y)
print("Shape of X :", X.shape)
print("Shape of y :", y.shape)
y = np.reshape(y,(y.shape[0],y.shape[1]*y.shape[2]))
print("Shape of y :", y.shape)


# ## **Creating Model**

input_img = Input(shape = (224,224,1))

z1 = Conv2D(32, (5,5))(input_img)
a1 = Activation('relu')(z1)
a1 = MaxPooling2D(pool_size=(2,2), strides=2)(a1)
a1 = Dropout(0.2)(a1)

z2 = Conv2D(32,(5,5))(a1)
a2 = Activation('relu')(z2)
a2 = MaxPooling2D(pool_size=(2,2), strides=2)(a2)
a2 = Dropout(0.2)(a2)

z3 = Conv2D(64,(5,5))(a2)
a3 = Activation('relu')(z3)
a3 = MaxPooling2D(pool_size=(2,2))(a3)
a3 = Dropout(0.2)(a3)

z4 = Conv2D(64,(3,3))(a3)
a4 = Activation('relu')(z4)
a4 = MaxPooling2D(pool_size=(2,2), strides=2)(a4)
a4 = Dropout(0.2)(a4)

z5 = Conv2D(128,(3,3))(a4)
a5 = Activation('relu')(z5)
a5 = MaxPooling2D(pool_size=(2,2), strides=2)(a5)
a5 = Dropout(0.2)(a5)

a5 = Flatten()(a5)

z6 = Dense(136)(a5)

# ----------------------------

model = Model(input_img, z6)
model.summary()


# Compile and train model

model.compile(optimizer=Adam(0.001), loss='mse')

train = model.fit(X, y, batch_size=64, epochs=45)


# ## **Plot loss**

def plot_history(history):
    fig = plt.figure(figsize=(15,8))
    ax2 = fig.add_subplot(222)
    ax2.set_title('model loss')
    ax2.plot(history['loss'])

    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper right')

plot_history(train.history)


# ## **Save model**

# In[ ]:


model.save('Model.h5')

# ## **Test Model**


def denormalize_keypoints(keypoints):
    keypoints = keypoints*50+100
    return keypoints


def preprocess_test(data):

    img_names = data['Unnamed: 0'].values
    X = []
    Y = []
    n = 0

    for file in img_names:
        img = cv2.imread('\\data\\test\\'+file)
        x,y,z = img.shape
        img = cv2.resize(img,(224,224))

        key_pts = data.iloc[n, 1:].to_numpy()
        key_pts = key_pts.astype('float').reshape(-1, 2)
        key_pts = key_pts * [224 / y, 224 / x]

        image_copy = np.copy(img)
        key_pts_copy = np.copy(key_pts)

        image_copy = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        image_copy=  image_copy/255.0

        X.append(image_copy)
        Y.append(key_pts_copy)

        n += 1

    return X,Y


def plot_predictions(X_test,y_test,predictions):
    show = 10
    for i in range(show):
      plt.figure(figsize=(5,25))
      ax = plt.subplot(1,show,i+1)
      image = X_test[i]
      p = predictions[i]
      y = y_test[i]

      plt.imshow(np.squeeze(image), cmap = 'gray')
      plt.scatter(p[:,0], p[:,1], s=20, marker='.', c='m')
      plt.scatter(y[:, 0], y[:, 1], s=20, marker='.', c='g')

    plt.show()


test_keypoints = pd.read_csv('\\data\\test_frames_keypoints.csv')


X_test,y_test = preprocess_test(test_keypoints)

X_test = np.array(X_test)
y_test = np.array(y_test)

X_test = np.expand_dims(X_test,axis = 3)



predictions = model.predict(X_test)
predictions.shape


predictions = np.reshape(predictions,(predictions.shape[0],int(predictions.shape[1]/2),2))

for i in range(predictions.shape[0]):
    predictions[i] = denormalize_keypoints(predictions[i])


plot_predictions(X_test,y_test,predictions)

