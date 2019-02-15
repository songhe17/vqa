import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,SequentialSampler,BatchSampler
import numpy as np
import pickle
from vqa.dataset import Dictionary, VQAFeatureDataset
import vqa.base_model as base_model
from vqa.train import train,evaluate
import vqa.utils
import tensorflow as tf
from tqdm import tqdm
from bleu import BLEU
with open('data/dictionary.pkl','rb') as f:
	dic = pickle.load(f)
	dic = dic[0]
	dictionaries = {}
	for i in dic:
		dictionaries[str(dic[i])] = i
def num2word(sentences,dictionary):
        sents = []
        for sent in sentences:
                words = []
                for word in sent:
                        if word != 19901 and word != 19903:
                                words.append(dictionary[str(word)])
                words = ' '.join(words)
                sents.append(words)

        return sents

def labeltoconfi(labels,confis):
    confidence = []
    for i in range(len(labels)):
	controller = False
        index = labels[i]
	if index != np.argmax(confis[i]):

	    controller = True
            #index = np.argmax(confis[i])
        confi = confis[i][index]
	if controller:
	    confi = 1.0
        confidence.append(confi)
    return confidence
def _reverse(sentences):
    sents = []
    for sent in sentences:
        rev = [word for word in sent if word < 19901]
        padding = [19901] * (14 - len(rev))
        res = padding + rev
        sents.append(res)

    return sents




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args


#eval_loader =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1)

def train(qg_model,sess):
	args = parse_args()

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.backends.cudnn.benchmark = True

	dictionary = Dictionary.load_from_file('vqa/data/dictionary.pkl')
	train_dset = VQAFeatureDataset('train', dictionary)
	#eval_dset = VQAFeatureDataset('val', dictionary)
	batch_size = args.batch_size

	constructor = 'build_%s' % args.model
	model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
	model.w_emb.init_embedding('vqa/data/glove6b_init_300d.npy')

	model = nn.DataParallel(model).cuda()
	model.load_state_dict(torch.load('vqa/saved_models/exp0/model.pth'))
	model.eval()
	config = qg_model.config
	_ = tf.Variable(initial_value='fake_variable')
	sess.run(tf.global_variables_initializer())
	var_filter_fn = lambda name: ("leanring_rate" not in name)
	vars_to_restore = [v for v in tf.global_variables() if var_filter_fn(v.name)]
	saver = tf.train.Saver(var_list=vars_to_restore)
	if config.reinforce:

		saver.restore(sess,qg_model.config.model_save_path)

	
	for _ in tqdm(list(range(config.num_epochs)), desc='epoch'):
		for _ in tqdm(list(range(config.num_batches)), desc='train_batch'):

			#batch = data.next_batch()
			#batch = data
			
			
			batch_questions,batch_answers,batch_feats,batch_masks,start,end,batch_anslabels = train_dset.qg_getitem(config.batch_size,False)
			batch = batch_questions,batch_answers,batch_feats,batch_masks
			#print(np.shape(batch_questions))
			#print(np.shape(batch_answers))
			#print(np.shape(batch_feats))
			#print(np.shape(batch_masks))
			#break




			if config.self_critical:

				batch_baseline = sc_evl(qg_model,batch,sess)

				sc_sents,batch_reward = evl(qg_model,batch,sess)

				feed_dict = {qg_model.contexts:batch_feats,
				qg_model.questions:batch_questions,
				qg_model.answers:batch_answers,
				qg_model.sc_sents:sc_sents,
				qg_model.sc_evl:batch_reward,
				qg_model.baseline:batch_baseline,
				qg_model.masks:batch_masks}

			if config.reinforce:

				predics = evl(qg_model,batch,sess)
				k = num2word(predics,dictionaries)
				print(k)	
				train_dset.reset_q(predics)

				sampler = BatchSampler(SequentialSampler(range(config.batch_size)),batch_size=30, drop_last=False)
				
				train_loader = DataLoader(train_dset,num_workers=0,batch_sampler = sampler)

				_,_,confidence = evaluate(model,train_loader)
				#confidence = np.zeros((batch_size,))
				batch_confidence = labeltoconfi(batch_anslabels,confidence)
				feed_dict = {qg_model.contexts:batch_feats,
				qg_model.questions:batch_questions,
				qg_model.answers:batch_answers,
				qg_model.confidence:batch_confidence,
				qg_model.masks:batch_masks}
			else:
				feed_dict = {qg_model.contexts:batch_feats,
				qg_model.questions:batch_questions,
				qg_model.answers:batch_answers,
				qg_model.masks:batch_masks}

			if config.reinforce:
				_,tloss,rl_total_loss,reward = sess.run([qg_model.rlml_opt_op,qg_model.total_loss,qg_model.rl_total_loss,qg_model.reward],feed_dict = feed_dict)
				print("RL loss:"+str(rl_total_loss))
				print('reward'+str(reward)) 
			else if config.self_critical:

			_,tloss = sess.run([qg_model.opt_op,qg_model.total_loss],feed_dict = feed_dict)
			print("Batch loss: "+str(tloss))
			#print('reward'+str(reward))			
		print("Total loss: "+str(tloss))		 
	if config.reinforce:

		saver.save(sess,qg_model.config.rl_model_save_path)
	else:
		saver.save(sess,qg_model.config.model_save_path)


def sc_evl(qg_model,batch,sess):
        #saver = tf.train.Saver()
        #saver.restore(sess,qg_model.config.rl_model_save_path)
        #config = qg_model.config

        batch_questions,batch_answers,batch_feats,batch_masks = batch

        feed_dict = {qg_model.contexts:batch_feats,
                qg_model.answers:batch_answers,
                qg_model.masks:batch_masks,
                qg_model.questions:batch_questions}
        config.is_train = False
        mpredictions,mquestions = sess.run([qg_model.predictions,qg_model.questions],feed_dict = feed_dict)
        mpredictions = np.array(mpredictions).transpose().tolist()
        config.is_train = True
        baseline = num2word(mpredictions,dictionaries)
        ques = num2word(batch_questions,dictionaries)
        ques = np.expand_dims(ques,0)
        score = BLEU(baseline,ques)
        return score

def evl(qg_model,batch,sess):
	#saver = tf.train.Saver()
	#saver.restore(sess,qg_model.config.rl_model_save_path)
	#config = qg_model.config
	score = []
	batch_questions,batch_answers,batch_feats,batch_masks = batch

	feed_dict = {qg_model.contexts:batch_feats,
		qg_model.answers:batch_answers,
		qg_model.masks:batch_masks,
		qg_model.questions:batch_questions}
	mpredictions,mquestions = sess.run([qg_model.rl_predictions,qg_model.questions],feed_dict = feed_dict)
	if config.reinforce:
		mpredictions = _reverse(np.array(mpredictions).transpose().tolist())
	else:
		mpredictions = np.array(mpredictions).transpose().tolist()
	if config.self_critical:
		baseline = num2word(mpredictions,dictionaries)
		ques = num2word(batch_questions,dictionaries)
        ques = np.expand_dims(ques,0)
        score = BLEU(baseline,ques)

	#if not config.is_train:
	#	with open('eval_q.pkl','wb') as f:
	#		pickle.dump(questions,f)
	#	with open('eval_prediction.pkl','wb') as f:
	#		pickle.dump(predictions,f)
	return mpredictions,score

def _evaluate(qg_model,sess):
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('vqa/data/dictionary.pkl')
    #train_dset = VQAFeatureDataset('train', dictionary)
    eval_dset = VQAFeatureDataset('val', dictionary)
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid).cuda()
    model.w_emb.init_embedding('vqa/data/glove6b_init_300d.npy')

    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('vqa/saved_models/exp0/model.pth'))
    model.eval()
    config = qg_model.config
    _ = tf.Variable(initial_value='fake_variable')
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess,qg_model.config.model_save_path)
    predictions = []
    questions = []
    for _ in tqdm(list(range(318)), desc='train_batch'):
	batch_questions,batch_answers,batch_feats,_,_,_ = eval_dset.qg_getitem(config.batch_size,True)
	feed_dict = {qg_model.contexts:batch_feats,
	qg_model.answers:batch_answers,
	qg_model.questions:batch_questions}
	mpredictions,mquestions = sess.run([qg_model.predictions,qg_model.questions],feed_dict = feed_dict)
	mpredictions = np.array(mpredictions).transpose().tolist()
	predictions.extend(mpredictions)
	questions.extend(mquestions)
    with open('q_1.pkl','wb') as f:
	pickle.dump(questions,f)
    with open('pred_1.pkl','wb') as f:
	pickle.dump(predictions,f)
    return predictions,questions

