#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 2019

@author: songhewang
"""
#dataset
import pickle
import numpy as np
class Dataset(object):

	def __init__(self,config):
		self.config = config
		self.load_data()

	def load_data(self):
		with open(self.config.feats_save_path,'rb') as f:#if config.is_train, then config.feats_save_path is training set
			self.feats = pickle.load(f)

		with open(self.config.ans_save_path,'rb') as f:
			self.answers = pickle.load(f)
			self.answers = self.answers[:len(self.feats)]

		with open(self.config.q_save_path,'rb') as f:
			self.questions = pickle.load(f)
			self.questions = self.questions[:len(self.feats)]

		with open(self.config.masks_save_path,'rb') as f:
			self.masks = pickle.load(f)
			self.masks = self.masks[:len(self.feats)]

		with open('data/train_ans2label.pkl','rb') as f:
			self.ans2labels = pickle.load(f)
			self.ans2labels = self.ans2labels[:len(self.feats)]

	def next_batch(self):
		config = self.config
		start = np.random.randint(len(self.questions) - config.batch_size)

		end = start + config.batch_size

		batch_feats = self.feats[start:end]

		batch_answers = self.answers[start:end]

		batch_questions = self.questions[start:end]

		batch_masks = self.masks[start:end]

		batch_anslabels = self.ans2labels[start:end]

		batch = (batch_questions,batch_answers,batch_feats,batch_masks,batch_anslabels,start,end)

		#self.batch_counter += self.config.batch_size

		return batch





