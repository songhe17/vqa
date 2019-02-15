#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 2019

@author: songhewang
"""
#config

class Config(object):

	def __init__(self):
		self.reinforce = True
		self.plan1 = False
		self.is_train = True
		self.dim_emb = 512
		self.ans_dim_emb = 300
		self.hidden_size = 512
		self.num_batches = 200
		#self.batch_size = 256
		self.num_ctx = 36
		self.dim_ctx = 2048
		self.max_caption_length = 14
		self.vocabulary_size = 19903
		self.num_initalize_layers = 2    # 1 or 2
        	self.dim_initalize_layer = 512
        	self.num_attend_layers = 2       # 1 or 2
        	self.at_middle_units = 512
        	self.num_decode_layers = 2       # 1 or 2
        	self.dim_decode_layer = 1024
        	self.lstm_drop_rate = 0.3
		self.attention_loss_factor = 0.01 
		#optimizer
		self.num_epochs = 20
        	self.batch_size = 32
        	self.optimizer = 'Adam'    # 'Adam', 'RMSProp', 'Momentum' or 'SGD'
        	self.initial_learning_rate = 0.0001
        	self.learning_rate_decay_factor = 1.0
        	self.num_steps_per_decay = 100000
        	self.clip_gradients = 5.0
        	self.momentum = 0.0
        	self.use_nesterov = True
        	self.decay = 0.9
        	self.centered = True
        	self.beta1 = 0.9
        	self.beta2 = 0.999
        	self.epsilon = 1e-4
		self.lamda = 0.99
        #save path
	       	self.model_save_path = 'model/models'
		self.rl_model_save_path = 'rl_model/rl_mlodel'
        	if self.is_train:
	        	self.feats_save_path = 'data/train_features.pkl'
	        	self.confi_save_path = 'data/train_confidence.pkl'
	        	self.ans_save_path = 'data/train_emb_ans.pkl'
	        	self.q_save_path = 'data/train_token_q.pkl'
	        	self.masks_save_path ='data/train_masks.pkl'
	    
		else:
	    		self.feats_save_path = 'data/val_features.pkl'
	        	#self.confi_save_path = 'data/train_confidence.pkl'
	        	self.ans_save_path = 'data/val_emb_ans.pkl'
	        	self.q_save_path = 'data/val_token_q.pkl'
	        	self.masks_save_path = 'data/val_masks.pkl'



