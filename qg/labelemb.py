
import pickle
import numpy as np
#import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from tqdm import tqdm
with open('data/dictionary.pkl','rb') as f:
    dictionary = pickle.load(f)
    word2label = dictionary[0]
    for key in word2label.keys():
        word2label[key] += 1
    word2label.update({'<start>':0})
    word2label.update({'<null>':len(word2label)})
with open('data/word2emb.pkl','rb') as f:
    word2emb = pickle.load(f)
with open('data/trainval_ans2label.pkl','rb') as f:
    das = pickle.load(f)

    das = sorted(das.items(), key=lambda kv: kv[1])
    #print(das)
def embed(ans,word2emb):
    answers = []
    for a in tqdm(ans,desc = 'ans_embed'):
        a = a[0]
        a = word_tokenize(a)
        emb = []
        for word in a:
            if word not in word2emb:
                word = 'unk'
                print(word)
            value = word2emb[word]
            emb.append(value)
        emb = np.mean(emb,axis=0)
        answers.append(emb)
    return answers
label_emb = embed(das,word2emb)
with open("data/label_emb.pkl",'rb') as f:
    pickle.dump(label_emb,f)
