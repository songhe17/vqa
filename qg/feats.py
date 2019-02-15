#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 2019

@author: songhewang
"""

'data preprocess'
import pickle
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm

with open('data/v2_OpenEnded_mscoco_train2014_questions.json','rb') as f:
    train_q = json.load(f)
    train_q = train_q['questions']
    train_q = sorted(train_q,key = lambda x:x['question_id'])
    print("Finish loading train questions")

with open('data/v2_OpenEnded_mscoco_val2014_questions.json','rb') as f:
    val_q = json.load(f)
    val_q = val_q['questions']
    val_q = sorted(val_q,key = lambda x:x['question_id'])
    print("Finish loading val annotations")



with open("data/feats.pkl", "r") as f:
    feats = pickle.load(f)

im_name_to_id = {}
with open('data/instances_train2014.json','rb') as f:
    data = json.load(f)
    images = data['images']
    for image in tqdm(images,desc = 'instances_train2014'):
        name = image['file_name']
        ids = image['id']
        im_name_to_id[name] = ids

with open('data/instances_val2014.json','rb') as f:
    datas = json.load(f)
    images = datas['images']
    for image in tqdm(images,desc = 'instances_val2014'):
        name = image['file_name']
        ids = image['id']
        im_name_to_id[name] = ids



def order_feats(qs,feats,im_name_to_id):
    features = {}
    img_feats = []
    for name in tqdm(feats.keys(),desc = 'order_feats_a'):
        feat = feats[name]
        print(np.shape(feat))
        ids = im_name_to_id[name]
        features[ids] = feat
    for q in tqdm(qs,desc = 'order_feats_b'):
        imid = q['image_id']
        feat = features[imid]
        img_feats.append(feat)
    return img_feats

if __name__ == '__main__':

    #answers

    #get the ordered features
    train_features = order_feats(train_q,feats,im_name_to_id)
    train_features = train_features[:100]
    train_features = np.reshape(train_features,(100,196,512))


    with open('try_train_features.pkl','wb') as f:
        pickle.dump(train_features,f)


    print("Finish processing features")
    #embed answers


    print("Finishing questions and masks")


 
            
            
            
            
