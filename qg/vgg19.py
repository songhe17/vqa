#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 2019

@author: songhewang
"""
import numpy as np
import tensorflow as tf
import keras
import os,os.path
import pickle
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import decode_predictions
from keras.applications.vgg19 import VGG19
class Encoder(object):

    def __init__(self, image_path,save_path,config):
        self.path = image_path
        self.save_path = save_path
        self.config = config
    def _extract_feats(self):
        model = VGG19(include_top=True, weights='imagenet')
        model.layers.pop() # Get rid of the classification layer
        model.layers.pop() # Get rid of the dropout layer
        model.outputs = [model.layers[-4].output]
        model.layers[-1].outbound_nodes = []
        path = self.path
        data = {}
        for filename in os.listdir(path):
            image = load_img(os.path.join(path,filename),target_size = (224,224))
            image = img_to_array(image)
            image = np.reshape(image,(1,image.shape[0],image.shape[1],image.shape[2]))
            image = preprocess_input(image)
            labels = model.predict(image)
            labels = np.reshape(labels,(1,196,512))
            #filename = filename.split('.')[0]
            data[filename] = labels
        with open(self.save_path,'wb') as f:
            pickle.dump(data,f)

file_path = self.config.image_path
save_path = self.config.feats_save_path
trythisone = encoder(file_path,save_path)
trythisone._extract_feats()
with open(save_path,'rb') as f:
    feats = pickle.load(f)
    print(np.shape(feats['fresh_nature.jpg']))

