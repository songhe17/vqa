#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 2019

@author: songhewang
"""
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import json
import pickle
#from config import Config  
class CaptionGenerator(object):
	def __init__(self,config):
		self.config = config
		if config.plan1:
			config.dim_emb = 812
			config.hidden_size = 812
		self.contexts = tf.placeholder(
			dtype = tf.float32,
			shape = [config.batch_size, config.num_ctx,config.dim_ctx])

		self.questions = tf.placeholder(
			dtype = tf.int32,
			shape = [config.batch_size, config.max_caption_length])

		self.answers = tf.placeholder(
			dtype = tf.float32,
			shape = [config.batch_size, config.ans_dim_emb])

		self.masks = tf.placeholder(
			dtype = tf.float32,
			shape = [config.batch_size, config.max_caption_length])
		
		self.sc_evl = tf.placeholder(
			dtype = tf.float32,
			shape = [config.batch_size])
		
		self.baseline = tf.placeholder(
			dtype = tf.float32,
			shape = [config.batch_size])
		

		if config.is_train and not config.reinforce:
			self.build_rnn()
			self.build_optimizer()

		if config.is_train and config.reinforce:
			self.confidence = tf.placeholder(
				dtype = tf.float32,
				shape = [config.batch_size])
			print(config.reinforce)
			self.build_rnn()
			self.rl_rnn()
			self.build_optimizer()

		if not config.is_train:
			self.build_rnn()
		
		if config.self_critical:
			self.

	def build_rnn(self):
		print("Building rnn")
		config = self.config
		self.kernel_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
		#set up placeholder
		contexts = self.contexts
		questions = self.questions
		answers = self.answers
		masks = self.masks
		#confidence = self.confidence

		with tf.variable_scope("word_embedding",reuse = tf.AUTO_REUSE):
			embedding_matrix = tf.get_variable(
				name = 'emb_weights',
				shape = [config.vocabulary_size, config.dim_emb],
				initializer = self.kernel_initializer,
				trainable = config.is_train)

		#LSTMcell

		lstm = tf.nn.rnn_cell.LSTMCell(num_units = config.hidden_size)
		if config.is_train:
			lstm = tf.nn.rnn_cell.DropoutWrapper(
				lstm,                
				input_keep_prob = 1.0-config.lstm_drop_rate,
				output_keep_prob = 1.0-config.lstm_drop_rate,
				state_keep_prob = 1.0-config.lstm_drop_rate)


		with tf.variable_scope("initialize",reuse = tf.AUTO_REUSE):

			initial_output,initial_memory = self.initialize(contexts)

		# Prepare to run
		predictions = []
		if config.is_train:
			alphas = []
			cross_entropies = []
			#predictions_correct = []
		num_steps = config.max_caption_length
		#last_output = initial_output
		#last_memory = initial_memory
		last_word = tf.convert_to_tensor([config.vocabulary_size - 1] * config.batch_size)
		last_output = initial_output
		last_memory = initial_memory
		last_state = last_memory, last_output
		# Generate the words one by one
		for idx in range(num_steps):
			# Attention mechanism
			with tf.variable_scope("attend",reuse = tf.AUTO_REUSE):
				alpha = self.attend(contexts, last_output)

				context = tf.reduce_sum(contexts*tf.expand_dims(alpha, 2),
										axis = 1)
				context = tf.concat([context,answers],1)


				if config.is_train:
					tiled_masks = tf.tile(tf.expand_dims(masks[:, idx], 1),
										 [1, config.num_ctx])
					masked_alpha = alpha * tiled_masks
					alphas.append(tf.reshape(masked_alpha, [-1]))

			with tf.variable_scope("word_embedding",reuse = tf.AUTO_REUSE):
				word_embed = tf.nn.embedding_lookup(embedding_matrix,
													last_word)
		   # Apply the LSTM
			with tf.variable_scope("lstm",reuse = tf.AUTO_REUSE):
				if not config.plan1:
					context = tf.layers.dense(
						inputs = context,
						units = config.dim_emb,
						name = 'plan1_mtx')
				current_input = tf.concat([context, word_embed], 1)


				output, state = lstm(current_input, last_state)


				memory, _ = state


			with tf.variable_scope("decode",reuse = tf.AUTO_REUSE):
				expanded_output = tf.concat([output,
											 context,
											 word_embed],
											 axis = 1)
				logits = self.decode(expanded_output)
				probs = tf.nn.softmax(logits)


				prediction = tf.argmax(logits, 1)
				predictions.append(prediction)


				last_output = output

				last_memory = memory

				last_state = state

			if config.is_train and not config.reinforce:
				cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
					labels = questions[:,idx],
					logits = logits)

				masked_cross_entropy = cross_entropy * masks[:,idx]
				cross_entropies.append(masked_cross_entropy)
				last_word = questions[:, idx]
			else if config.is_train and config.reinforce:
				last_word = prediction


			tf.get_variable_scope().reuse_variables()

		if config.is_train:
			cross_entropies = tf.stack(cross_entropies, axis = 1)
			cross_entropy_loss = tf.reduce_sum(cross_entropies)/tf.reduce_sum(masks)

			alphas = tf.stack(alphas, axis = 1)
			alphas = tf.reshape(alphas, [config.batch_size, config.num_ctx, -1])
			attentions = tf.reduce_sum(alphas, axis = 2)
			diffs = tf.ones_like(attentions) - attentions
			attention_loss = config.attention_loss_factor*tf.nn.l2_loss(diffs)/(config.batch_size * config.num_ctx)

			reg_loss = tf.losses.get_regularization_loss()

			total_loss = cross_entropy_loss + attention_loss + reg_loss

		#self.contexts = contexts
		if config.is_train:
			#self.questions = questions
			#self.masks = masks
			self.total_loss = total_loss
			self.cross_entrspy_loss = cross_entropy_loss
			self.attention_loss = attention_loss
			self.reg_loss = reg_loss
			self.attentions = attentions
		else:
			self.initial_memory = initial_memory
			self.initial_output = initial_output
			self.last_memory = last_memory
			self.last_output = last_output
			self.last_word = last_word
			self.memory = memory
			self.output = output
			self.probs = probs
			self.predictions = predictions

	def rl_rnn(self):
		print("building rl rnn")
		config = self.config
		self.kernel_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
		#set up placeholder
		contexts = self.contexts
		questions = self.questions
		answers = self.answers
		masks = self.masks
		if config.is_train:
			confidence = self.confidence

		with tf.variable_scope("word_embedding",reuse = tf.AUTO_REUSE):
			embedding_matrix = tf.get_variable(
				name = 'emb_weights',
				shape = [config.vocabulary_size, config.dim_emb],
				initializer = self.kernel_initializer,
				trainable = config.is_train)

		#LSTMcell

		lstm = tf.nn.rnn_cell.LSTMCell(num_units = config.hidden_size)
		if config.is_train:
			lstm = tf.nn.rnn_cell.DropoutWrapper(
				lstm,                
				input_keep_prob = 1.0-config.lstm_drop_rate,
				output_keep_prob = 1.0-config.lstm_drop_rate,
				state_keep_prob = 1.0-config.lstm_drop_rate)


		with tf.variable_scope("initialize",reuse = tf.AUTO_REUSE):

			initial_output,initial_memory = self.initialize(contexts)

		# Prepare to run
		predictions = []
		if config.is_train:
			alphas = []
			cross_entropies = []
			predictions = []
			num_steps = config.max_caption_length
			last_output = initial_output
			last_memory = initial_memory
			last_word = tf.convert_to_tensor([config.vocabulary_size - 1] * config.batch_size)
		last_state = last_memory, last_output

		# Generate the words one by one
		for idx in range(num_steps):
			# Attention mechanism
			with tf.variable_scope("attend",reuse = tf.AUTO_REUSE):
				alpha = self.attend(contexts, last_output)
				context = tf.reduce_sum(contexts*tf.expand_dims(alpha, 2),
										axis = 1)
				context = tf.concat([context,answers],1)


				tiled_masks = tf.tile(tf.expand_dims(masks[:, idx], 1),
										 [1, config.num_ctx])
				masked_alpha = alpha * tiled_masks
				alphas.append(tf.reshape(masked_alpha, [-1]))

			with tf.variable_scope("word_embedding",reuse = tf.AUTO_REUSE):
				word_embed = tf.nn.embedding_lookup(embedding_matrix,
													last_word)
		   # Apply the LSTM
			with tf.variable_scope("lstm",reuse = tf.AUTO_REUSE):
				if not config.plan1:
					context = tf.layers.dense(
						inputs = context,
						units = config.dim_emb,
						name = 'plan1_mtx')
				current_input = tf.concat([context, word_embed], 1)
				output, state = lstm(current_input, last_state)
				memory, _ = state

			with tf.variable_scope("decode",reuse = tf.AUTO_REUSE):
				expanded_output = tf.concat([output,
											 context,
											 word_embed],
											 axis = 1)
				logits = self.decode(expanded_output)
				probs = tf.nn.softmax(logits)
				#prediction = tf.argmax(logits, 1)
                                prediction = tf.multinomial(logits,1)
                                prediction = tf.reshape(prediction,[-1])
				predictions.append(prediction)

			if config.is_train:
				#prediction = tf.multinomial(logits,1)
                                #prediction = tf.reshape(prediction,[-1])
				cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
					labels = questions[:,idx],
					logits = logits)
				masked_cross_entropy = cross_entropy * masks[:,idx]
				#mean = tf.reduce_mean(confidence)
				controller = tf.reduce_mean(confidence)
				self.reward = -tf.log(confidence) * (1/10) 
				masked_cross_entropy = masked_cross_entropy * (self.reward)
				cross_entropies.append(masked_cross_entropy)
				last_output = output
				last_memory = memory
				last_state = state
				last_word = prediction

				tf.get_variable_scope().reuse_variables()

		self.rl_predictions = predictions

		if config.is_train:
			cross_entropies = tf.stack(cross_entropies, axis = 1)
			cross_entropy_loss = tf.reduce_sum(cross_entropies) / tf.reduce_sum(masks)

			alphas = tf.stack(alphas, axis = 1)
			alphas = tf.reshape(alphas, [config.batch_size, config.num_ctx, -1])
			attentions = tf.reduce_sum(alphas, axis = 2)
			diffs = tf.ones_like(attentions) - attentions
			attention_loss = config.attention_loss_factor*tf.nn.l2_loss(diffs)/(config.batch_size * config.num_ctx)

			reg_loss = tf.losses.get_regularization_loss()

			total_loss = cross_entropy_loss + attention_loss + reg_loss


			self.rl_total_loss = total_loss
			self.rl_cross_entrspy_loss = cross_entropy_loss
			self.rl_attention_loss = attention_loss
			self.rl_reg_loss = reg_loss
			self.rl_attentions = attentions

	def sc_rnn(self):
		print("building sc rnn")
		config = self.config
		self.kernel_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
		#set up placeholder
		contexts = self.contexts
		questions = self.questions
		answers = self.answers
		masks = self.masks
		if config.is_train:
			confidence = self.confidence

		with tf.variable_scope("word_embedding",reuse = tf.AUTO_REUSE):
			embedding_matrix = tf.get_variable(
				name = 'emb_weights',
				shape = [config.vocabulary_size, config.dim_emb],
				initializer = self.kernel_initializer,
				trainable = config.is_train)

		#LSTMcell

		lstm = tf.nn.rnn_cell.LSTMCell(num_units = config.hidden_size)
		if config.is_train:
			lstm = tf.nn.rnn_cell.DropoutWrapper(
				lstm,                
				input_keep_prob = 1.0-config.lstm_drop_rate,
				output_keep_prob = 1.0-config.lstm_drop_rate,
				state_keep_prob = 1.0-config.lstm_drop_rate)


		with tf.variable_scope("initialize",reuse = tf.AUTO_REUSE):

			initial_output,initial_memory = self.initialize(contexts)

		# Prepare to run
		predictions = []
		if config.is_train:
			alphas = []
			cross_entropies = []
			predictions = []
			num_steps = config.max_caption_length
			last_output = initial_output
			last_memory = initial_memory
			last_word = tf.convert_to_tensor([config.vocabulary_size - 1] * config.batch_size)
		last_state = last_memory, last_output

		# Generate the words one by one
		for idx in range(num_steps):
			# Attention mechanism
			with tf.variable_scope("attend",reuse = tf.AUTO_REUSE):
				alpha = self.attend(contexts, last_output)
				context = tf.reduce_sum(contexts*tf.expand_dims(alpha, 2),
										axis = 1)
				context = tf.concat([context,answers],1)


				tiled_masks = tf.tile(tf.expand_dims(masks[:, idx], 1),
										 [1, config.num_ctx])
				masked_alpha = alpha * tiled_masks
				alphas.append(tf.reshape(masked_alpha, [-1]))

			with tf.variable_scope("word_embedding",reuse = tf.AUTO_REUSE):
				word_embed = tf.nn.embedding_lookup(embedding_matrix,
													last_word)
		   # Apply the LSTM
			with tf.variable_scope("lstm",reuse = tf.AUTO_REUSE):
				if not config.plan1:
					context = tf.layers.dense(
						inputs = context,
						units = config.dim_emb,
						name = 'plan1_mtx')
				current_input = tf.concat([context, word_embed], 1)
				output, state = lstm(current_input, last_state)
				memory, _ = state

			with tf.variable_scope("decode",reuse = tf.AUTO_REUSE):
				expanded_output = tf.concat([output,
											 context,
											 word_embed],
											 axis = 1)
				logits = self.decode(expanded_output)
				probs = tf.nn.softmax(logits)
				#prediction = tf.argmax(logits, 1)
                                prediction = tf.multinomial(logits,1)
                                prediction = tf.reshape(prediction,[-1])
				predictions.append(prediction)

			if config.is_train:
				#prediction = tf.multinomial(logits,1)
                                #prediction = tf.reshape(prediction,[-1])
				cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
					labels = questions[:,idx],
					logits = logits)
				masked_cross_entropy = cross_entropy * masks[:,idx]
				#mean = tf.reduce_mean(confidence)
				controller = tf.reduce_mean(confidence)
				self.reward = self.sc_evl-self.baseline
				masked_cross_entropy = masked_cross_entropy * (self.reward)
				cross_entropies.append(masked_cross_entropy)
			last_output = output
			last_memory = memory
			last_state = state
			last_word = prediction

				tf.get_variable_scope().reuse_variables()

		self.sc_predictions = predictions

		if config.is_train:
			cross_entropies = tf.stack(cross_entropies, axis = 1)
			cross_entropy_loss = tf.reduce_sum(cross_entropies) / tf.reduce_sum(masks)

			alphas = tf.stack(alphas, axis = 1)
			alphas = tf.reshape(alphas, [config.batch_size, config.num_ctx, -1])
			attentions = tf.reduce_sum(alphas, axis = 2)
			diffs = tf.ones_like(attentions) - attentions
			attention_loss = config.attention_loss_factor*tf.nn.l2_loss(diffs)/(config.batch_size * config.num_ctx)

			reg_loss = tf.losses.get_regularization_loss()

			total_loss = cross_entropy_loss + attention_loss + reg_loss


			self.sc_total_loss = total_loss
			self.sc_cross_entrspy_loss = cross_entropy_loss
			self.sc_attention_loss = attention_loss
			self.sc_reg_loss = reg_loss
			self.sc_attentions = attentions


	def initialize(self,contexts):

		context_mean = tf.reduce_mean(contexts, axis = 1)

		config = self.config

		context_mean = tf.layers.dropout(context_mean)

		if config.num_initalize_layers == 1:
			# use 1 fc layer to initialize
			memory = tf.layers.dense(inputs = context_mean,
								   units = config.hidden_size,
								   activation = None,
								   name = 'fc_a')
			output = tf.layers.dense(inputs = context_mean,
								   units = config.hidden_size,
								   activation = None,
								   name = 'fc_b')
		else:
			# use 2 fc layers to initialize
			temp1 = tf.layers.dense(inputs = context_mean,
								  units = config.dim_initalize_layer,
								  activation = tf.tanh,
								  name = 'fc_a1')
			temp1 = tf.layers.dropout(temp1)
			memory = tf.layers.dense(inputs = temp1,
								   units = config.hidden_size,
								   activation = None,
								   name = 'fc_a2')

			temp2 = tf.layers.dense(inputs = context_mean,
								  units = config.dim_initalize_layer,
								  activation = tf.tanh,
								  name = 'fc_b1')
			temp2 = tf.layers.dropout(temp2)
			output = tf.layers.dense(inputs = temp2,
								   units = config.hidden_size,
								   activation = None,
								   name = 'fc_b2')
		return memory, output
		

	def attend(self,contexts,outputs):

		config = self.config

		reshaped_context = tf.reshape(contexts,[-1,config.dim_ctx])

		reshaped_context = tf.layers.dropout(reshaped_context)

		outputs = tf.layers.dropout(outputs)

		if config.num_attend_layers == 1:

			temp1 = tf.layers.dense(
				inputs = reshaped_context,
				units = 1,
				use_bias = False,
				name = 'at_a')

			temp1 = tf.reshape(temp1,[-1,config.num_ctx])

			temp2 = tf.layers.dense(
				inputs = outputs,
				units = config.num_ctx,
				use_bia = False,
				name = 'at_b')
			logit = temp1 + temp2

		else:

			temp1 = tf.layers.dense(
				inputs = reshaped_context,
				units = config.at_middle_units,
				activation = tf.tanh,
				name = 'at_a1')

			temp2 = tf.layers.dense(
				inputs = outputs,
				units = config.at_middle_units,
				activation = tf.tanh,
				name = 'at_a2')

			temp2 = tf.tile(tf.expand_dims(temp2,1),[1,config.num_ctx,1])
			temp2 = tf.reshape(temp2,[-1,config.at_middle_units])

			temp = temp1 + temp2

			temp = tf.layers.dropout(temp)

			logits = tf.layers.dense(inputs = temp,
								   units = 1,
								   activation = None,
								   use_bias = False,
								   name = 'fc_2')
			logits = tf.reshape(logits, [-1, config.num_ctx])
		alpha = tf.nn.softmax(logits)
		return alpha

	def decode(self, expanded_output):
		""" Decode the expanded output of the LSTM into a word. """
		config = self.config
		expanded_output = tf.layers.dropout(expanded_output)
		if config.num_decode_layers == 1:
			# use 1 fc layer to decode
			logits = tf.layers.dense(expanded_output,
								   units = config.vocabulary_size,
								   activation = None,
								   name = 'fc')
		else:
			# use 2 fc layers to decode
			temp = tf.layers.dense(expanded_output,
								 units = config.dim_decode_layer,
								 activation = tf.tanh,
								 name = 'fc_1')
			temp = tf.layers.dropout(temp)
			logits = tf.layers.dense(temp,
								   units = config.vocabulary_size,
								   activation = None,
								   name = 'fc_2')
		return logits

	def build_optimizer(self):
		""" Setup the optimizer and training operation. """
		config = self.config

		learning_rate = tf.constant(config.initial_learning_rate)
		if config.learning_rate_decay_factor < 1.0:
			def _learning_rate_decay_fn(learning_rate, global_step):
				return tf.train.exponential_decay(
					learning_rate,
					global_step,
					decay_steps = config.num_steps_per_decay,
					decay_rate = config.learning_rate_decay_factor,
					staircase = True)
			learning_rate_decay_fn = _learning_rate_decay_fn
		else:
			learning_rate_decay_fn = None

		with tf.variable_scope('optimizer', reuse = tf.AUTO_REUSE):
			if config.optimizer == 'Adam':
				optimizer = tf.train.AdamOptimizer(
					learning_rate = config.initial_learning_rate,
					beta1 = config.beta1,
					beta2 = config.beta2,
					epsilon = config.epsilon
					)
			elif config.optimizer == 'RMSProp':
				optimizer = tf.train.RMSPropOptimizer(
					learning_rate = config.initial_learning_rate,
					decay = config.decay,
					momentum = config.momentum,
					centered = config.centered,
					epsilon = config.epsilon
				)
			elif config.optimizer == 'Momentum':
				optimizer = tf.train.MomentumOptimizer(
					learning_rate = config.initial_learning_rate,
					momentum = config.momentum,
					use_nesterov = config.use_nesterov
				)
			else:
				optimizer = tf.train.GradientDescentOptimizer(
					learning_rate = config.initial_learning_rate
				)
			if not config.reinforce:
				opt_op = tf.contrib.layers.optimize_loss(
					loss = self.total_loss,
					global_step = tf.train.create_global_step(),
					learning_rate = learning_rate,
					optimizer = optimizer,
					clip_gradients = config.clip_gradients,
					learning_rate_decay_fn = learning_rate_decay_fn)
				self.opt_op = opt_op
			if config.reinforce:
				print(config.reinforce)
				rlml_opt_op = tf.contrib.layers.optimize_loss(
					loss = (1-config.lamda)*self.total_loss+config.lamda*self.rl_total_loss,
					global_step = tf.train.create_global_step(),
					learning_rate = learning_rate,
					optimizer = optimizer,
					clip_gradients = config.clip_gradients,
					learning_rate_decay_fn = learning_rate_decay_fn)
				self.rlml_opt_op = rlml_opt_op
			if config.self_critical:
				sc_opt_op = tf.contrib.layers.optimize_loss(
					loss = (1-config.lamda)*self.total_loss+config.lamda*self.sc_total_loss,
					global_step = tf.train.create_global_step(),
					learning_rate = learning_rate,
					optimizer = optimizer,
					clip_gradients = config.clip_gradients,
					learning_rate_decay_fn = learning_rate_decay_fn)
				self.sc_opt_op = sc_opt_op
		#self.opt_op = opt_op
		



#config = Config()
#caps  = CaptionGenerator(config)





		
		
