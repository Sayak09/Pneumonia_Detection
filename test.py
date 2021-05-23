# -*- coding: utf-8 -*-
"""
Created on Sun May 23 18:00:46 2021

@author: sayak
"""

import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing import image
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical

import datetime

model=load_model('best_weights_cnn.hdf5')



imagePathscheck=list(paths.list_images('check'))
datacheck=[]
labels=[]

for x in imagePathscheck:
    label=x.split(os.path.sep)[-2]
    labels.append(label)
    image=load_img(x,target_size=(150,150))
    image=img_to_array(image)
    image=preprocess_input(image)
    datacheck.append(image)
    
datacheck=np.array(datacheck,dtype="float32")
labels=np.array(labels)
    
lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

preds = model.predict(datacheck)

pred_op_check=np.round(preds)

from sklearn.metrics import accuracy_score, confusion_matrix

acc = accuracy_score(labels, pred_op_check)*100