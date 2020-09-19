#!/usr/bin/env python
# coding: utf-8

# In[1]:

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2


# In[2]:

key_points = pd.read_csv('data/training_frames_keypoints.csv')
key_points.head()


# In[3]:

n = 0 
image_name = key_points.iloc[n,0]
key = key_points.iloc[n, 1:].to_numpy()
key = key.astype('float').reshape(-1,2)

print('Image name: ', image_name)
print('Landmarks shape: ',key.shape)
print('First 4 key points: {}'.format(key[:4]))


print("Number of images: ",key_points.shape[0])


# In[4]:

def show_keypoints(image, key):
    plt.imshow(image)
    plt.scatter(key[:,0],key[:,1], s=20,marker='.',c = 'm')

n = 300

image_name = key_points.iloc[n,0]
key = key_points.iloc[n, 1:].to_numpy()
key = key.astype('float').reshape(-1,2)

plt.figure(figsize=(5,5))
show_keypoints(mpimg.imread(os.path.join('data/training/', image_name)), key)
plt.show()


# In[5]:

# Example for Resizing Image and Resizing keypoints
n = 300
image_name = key_points.iloc[n,0]

img = cv2.imread('data/training/'+image_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
x,y,z = img.shape
img = cv2.resize(img, (224,224))

key = key_points.iloc[n, 1:].to_numpy()
key = key.astype('float').reshape(-1,2)

key = key *[224/y,224/x]


# In[6]:

show_keypoints(img, key)

