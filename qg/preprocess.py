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
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
tokenizer = RegexpTokenizer(r'\w+')
with open('data/v2_mscoco_train2014_annotations.json','rb') as f:
    train_ano = json.load(f)
    train_ano = train_ano['annotations']
    train_ano = sorted(train_ano,key = lambda x:x['question_id'])
    print("Finish loading train annotations")

with open('data/v2_mscoco_val2014_annotations.json','rb') as f:
    val_ano = json.load(f)
    val_ano = val_ano['annotations']
    val_ano = sorted(val_ano,key=lambda x:x['question_id'])
    print("Finish loading val annotations")

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

with open('data/trainval_ans2label.pkl','rb') as f:
    ans2label = pickle.load(f)
    print("Finish loading answer to labels")

with open("data/train_ans_confi.pkl","rb") as f:
    train_ans_confi = pickle.load(f)
    #train_ans_confi = train_ans_confis[0]
    #for i in tqdm(range(len(train_ans_confis)-1),desc = 'reshaping confidence'):
	#train_ans_confi = np.concatenate((train_ans_confi,train_ans_confis[i+1]))
    print("Finish loading train confidence")

with open("data/feats.pkl",'rb') as f:
    feats = pickle.load(f)
    print("Finish loading feats")

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

print("Finish creating image name to id")

with open('data/word2emb.pkl','rb') as f:
    word2emb = pickle.load(f)
print("Finish loading word2emb")

with open('data/dictionary.pkl','rb') as f:
    dictionary = pickle.load(f)
    word2label = dictionary[0]
    for key in tqdm(word2label.keys(),desc = 'dictionary'):
        word2label[key] += 3
    word2label.update({'<start>':0})
    word2label.update({'<null>':1})
    word2label.update({'<unk>':2})
print("Finish loading dictionary")

def order_feats(qs,feats,im_name_to_id):
    features = {}
    img_feats = []
    counter = 0
    for name in tqdm(feats.keys(),desc = 'order_feats_a'):
        feat = feats[name]
	if counter%1000 == 0:
		print(np.shape(feat))
        ids = im_name_to_id[name]
        features[ids] = feat
	counter += 1
    for q in tqdm(qs,desc = 'order_feats_b'):
        imid = q['image_id']
        feat = features[imid]
        img_feats.append(feat)
    return img_feats

def find_ans(annotations):
    answers = []
    for ano in tqdm(annotations,desc = 'find_ans'):
        l = 0
        for answer in ano['answers']:
            if answer['answer_confidence'] == 'yes':
                answers.append(answer['answer'])
                l = 1
                break
        if l == 0:
            for answer in ano['answers']:
                if answer['answer_confidence'] == 'maybe':
                    answers.append(answer['answer'])
                    l = 1
                    break
        if l == 0:
            for answer in ano['answers']:
                if answer['answer_confidence'] == 'no':
                    answers.append(answer['answer'])
                    l = 1
                    break
    return answers




def anstolabel(ans,al):
    anstolabel = []
    for answer in tqdm(ans,desc = 'anstolabel'):
        if answer in al.keys():
            anstolabel.append(str(al[answer]))
        else:
            anstolabel.append("argmax")
            
    return anstolabel


def labeltoconfi(labels,confis):
    confidence = []
    for i in range(len(labels)):
        if labels[i] != 'argmax':
            index = int(labels[i])
        else:
            index = np.argmax(confis[i])
        confi = confis[i][index]
        confidence.append(confi)
    return confidence       
def embed(ans,word2emb):
    answers = []
    for a in tqdm(ans,desc = 'ans_embed'):
        a = tokenizer.tokenize(a)
        emb = []
        for word in a:
            if word not in word2emb:
                word = 'unk'
            value = word2emb[word]
            emb.append(value)
	if len(emb) == 0:
	    emb.append(word2emb['unk'])
        emb = np.mean(emb,axis=0)
        answers.append(emb)
    return answers


def q_token(questions,dictionary):
    qs = []
    masks = []
    for q in tqdm(questions,desc = 'q_token'):
        q = q['question']
        mask = []
        q = tokenizer.tokenize(q.lower())
        if len(q) > 11:
            q = q[:11]
        q.append('<null>')
        while len(q) != 12:
            q.append('<null>')

        for i in range(len(q)):
            if q[i] != '<null>':
                mask.append(1.0)
            else:
                mask.append(0.0)
            if q[i] not in dictionary:
                label = 2
            else:
                label = dictionary[q[i]]
            q[i] = label
        masks.append(mask)
        qs.append(q)
        
    return qs,masks
if __name__ == '__main__':

    #answers
    train_ans = find_ans(train_ano)
    val_ans = find_ans(val_ano)

    print("Finish finding answers")
    #the label of answers
    train_anslabel = anstolabel(train_ans,ans2label)
    val_anslabel = anstolabel(val_ans,ans2label)

    print("Finish answers to labels")
    #confidence,numpy list
    train_confis = labeltoconfi(train_anslabel,train_ans_confi)
                
    #concatenate
    #questions = np.concatenate((train_q,val_q))
    #confidence = np.concatenate((train_confis,val_confis))
    #answers = np.concatenate((train_ans,val_ans))
    print("Finish processing labels")
    #get the ordered features
    train_features = order_feats(train_q,feats,im_name_to_id)
    val_features = order_feats(val_q,feats,im_name_to_id)

    print("Finish processing features")
    #embed answers
    train_emb_ans = embed(train_ans,word2emb)
    val_emb_ans = embed(val_ans,word2emb)

    print("Finish embedding answers")
    #token questions
    train_token_questions,train_masks = q_token(train_q,word2label)
    val_token_questions,val_masks = q_token(val_q,word2label)
    
    print("Finishing questions and masks")
    train_features = train_features[:10000]
    train_features = np.reshape(train_features,(10000,196,512))
    val_features = val_features[:1000]
    val_features = np.reshape(val_features,(1000,196,512))
    with open('train_confidence.pkl','wb') as f:
        pickle.dump(train_confis,f)

    with open('train_features.pkl','wb') as f:
        pickle.dump(train_features,f)

    with open('val_features.pkl','wb') as f:
        pickle.dump(val_features,f)

    with open('train_emb_ans.pkl','wb') as f:
        pickle.dump(train_emb_ans,f)

    with open('val_emb_ans.pkl','wb') as f:
        pickle.dump(val_emb_ans,f)

    with open('train_token_q.pkl','wb') as f:
        pickle.dump(train_token_questions,f)

    with open('val_token_q.pkl','wb') as f:
        pickle.dump(val_token_questions,f)
                
    with open('train_masks.pkl','wb') as f:
        pickle.dump(train_masks,f)

    with open('val_masks.pkl','wb') as f:
        pickle.dump(val_masks,f)
    
    print("ALL DONE!")
            
            
            
            
