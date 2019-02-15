#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 2019

@author: songhewang
"""
#main
print(1)
import numpy as np
import tensorflow as tf
from config import Config
from qg_model import CaptionGenerator
from dataset import Dataset
#from vgg19 import Encoder
from train_val import train,evaluate
from tensorflow.python.framework import ops
from nlgeval import NLGEval
ops.reset_default_graph()



def num2word(sentences,dictionary)
	sents = []
	for sent in sentences:
		words = []
		for word in sent:
			if word != 19901:
				words.append(dictionary[str(word)])
		words = ' '.join(words)
		sents.append(words)

	return sents
with open('data/dictionary.pkl','rb') as f:
	dic = pickle.load(f)
	dic = dic[0]
	dictionary = {}
	for i in dic:
		dictionary[str(dic[i])] = i
pred_words = num2word(predictions,dictionary)
q_words = num2word(questions,dictionary)
scores = n.compute_metrics(q_words,pred_words)
print(scores)
def main():
	FLAGS = tf.app.flags.FLAGS
	tf.flags.DEFINE_boolean('is_train', False,
	                       'True for training False for evaluation')

	tf.flags.DEFINE_boolean('plan1', False,
	                       'True for change the dimension to 812 False for ')

	tf.flags.DEFINE_boolean('reinforce', False,
	                       'Use mixed loss or not')
	config = Config()
	config.is_train = FLAGS.is_train
	config.reinforce = FLAGS.reinforce
	config.plan1 = FLAGS.plan1
	n = NLGEval()
	with tf.Session() as sess:
		fake_contexts = np.ones((config.batch_size, config.num_ctx,config.dim_ctx))
		fake_questions = np.ones((config.batch_size, config.max_caption_length))
		fake_answers = np.ones((config.batch_size, config.ans_dim_emb))
		fake_masks = np.ones((config.batch_size, config.max_caption_length))
		fake_confidence = np.ones(config.batch_size)
	
		data = Dataset(config)
		print(2)
		#data = (fake_questions,fake_answers,fake_contexts,fake_confidence,fake_masks)
		model = CaptionGenerator(config)
		#sess.run(tf.global_variables_initializer())
		if config.is_train:
			train(model,sess)
		else:
			predictions,questions = evaluate(model,sess)
			with open('data/dictionary.pkl','rb') as f:
				dic = pickle.load(f)
				dic = dic[0]
				dictionary = {}
				for i in dic:
					dictionary[str(dic[i])] = i
			pred_words = num2word(predictions,dictionary)
			q_words = num2word(questions,dictionary)
			scores = n.compute_metrics(q_words,pred_words)
			print(scores)

			


if __name__ == '__main__':
	main()
