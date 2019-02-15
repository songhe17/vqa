"""
reated on Tue Jan 15 2019

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


with open('data/dictionary.pkl','rb') as f:
    dictionary = pickle.load(f)
    word2label = dictionary[0]
    for key in tqdm(word2label.keys(),desc = 'dictionary'):
        word2label[key] += 3
    word2label.update({'<start>':0})
    word2label.update({'<null>':1})
    word2label.update({'<unk>':2})
print("Finish loading dictionary")



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
            masks.append(mask)#wrong
        qs.append(q)
        
    return qs,masks
if __name__ == '__main__':



    print("Finish embedding answers")
    #token questions
    train_token_questions,train_masks = q_token(train_q,word2label)
    val_token_questions,val_masks = q_token(val_q,word2label)

    print("Finishing questions and masks")



    with open('data/train_token_q.pkl','wb') as f:
        pickle.dump(train_token_questions,f)

    with open('data/val_token_q.pkl','wb') as f:
        pickle.dump(val_token_questions,f)
                
    with open('data/train_masks.pkl','wb') as f:
        pickle.dump(train_masks,f)

    with open('data/val_masks.pkl','wb') as f:
        pickle.dump(val_masks,f)
    
    print("ALL DONE!")
            
            
            
            
