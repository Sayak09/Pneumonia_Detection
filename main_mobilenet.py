# -*- coding: utf-8 -*-
"""
Created on Tue May 18 02:24:12 2021

@author: sayak
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 17 16:28:02 2021

@author: sayak
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.utils import plot_model
from PIL import Image
# General libraries

import pandas as pd 
import random
import cv2

# Deep learning libraries
#import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation,Conv2D

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf

imagePaths=list(paths.list_images('dataset'))
data=[]
labels=[]

for x in imagePaths:
    label=x.split(os.path.sep)[-2]
    labels.append(label)
    image=load_img(x,target_size=(150,150))
    image=img_to_array(image)
    image=preprocess_input(image)
    data.append(image)
    
data_new=np.array(data,dtype="float32")
labels=np.array(labels)
    
lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

X_train,X_test,y_train,y_test=train_test_split(data_new,labels,test_size=0.15,stratify=labels,random_state=40)

aug=ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

baseModel= MobileNetV2(weights="imagenet",include_top=False,input_shape=(150,150,3))

headModel=baseModel.output
headModel=AveragePooling2D(pool_size=(3,3))(headModel)
headModel=Flatten(name="flatten")(headModel)
headModel=Dense(128,activation="relu")(headModel)
headModel=Dropout(0.5)(headModel)
headModel=Dense(2,activation="softmax")(headModel)

model=Model(inputs=baseModel.input,outputs=headModel)
plot_model(model)

for layer in baseModel.layers:
    layer.trainable=False

checkpoint = ModelCheckpoint(filepath='best_weights_transfer_imagenet_mobilenetv2.hdf5', save_best_only=True, save_weights_only=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

EPOCHS=20
BS=32

H=model.fit(
    aug.flow(X_train,y_train,batch_size=BS),
    steps_per_epoch=len(X_train)/BS,
    validation_data=(X_test,y_test),
    validation_steps=len(X_test)/BS,
    epochs=EPOCHS,
    callbacks=[checkpoint, lr_reduce])

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax = ax.ravel()

for i, met in enumerate(['accuracy', 'loss']):
    ax[i].plot(H.history[met])
    ax[i].plot(H.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])
    
from sklearn.metrics import accuracy_score, confusion_matrix

preds = model.predict(X_test)
op_label=0

pred_op=np.round(preds)

acc = accuracy_score(y_test, np.round(preds))*100



    
    

