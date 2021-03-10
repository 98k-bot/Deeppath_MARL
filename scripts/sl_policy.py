from __future__ import division
from __future__ import print_function
import tensorflow as tf 
import numpy as np
from itertools import count
import math
import sys

from networks import policy_nn,policy_nn2,policy_nn_ad,policy_nn_ad2,q_network,q_network_old
from utils import *
from env import Env
from BFS.KB import KB
from BFS.BFS import BFS
import time

from utils import teacher
import multiprocessing
from multiprocessing import Pool
import random
from numpy import dot
from numpy.linalg import norm

output_state_test=[]

dataPath = "/home/huixin/DeepPath/NELL-995/"
relation = "concept_athletehomestadium"
# episodes = int(sys.argv[2])
graphpath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'

METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.25),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][0]        # choose the method for optimization



class SupervisedPolicy(object):
	"""docstring for SupervisedPolicy"""
	def __init__(self, learning_rate = 0.001):
		self.initializer = tf.contrib.layers.xavier_initializer()
		with tf.variable_scope('supervised_policy', reuse=tf.AUTO_REUSE):
			self.tflam = tf.placeholder(tf.float32, None, 'alpha')
			#self.kl = tf.placeholder(tf.float32, None, 'kl')
			self.state = tf.placeholder(tf.float32, [None, state_dim], name = 'state')
			self.action = tf.placeholder(tf.int32, [None], name = 'action')
			#self.action_prob = policy_nn_ad(self.state, self.action, state_dim, action_space, self.initializer,trainable=True)
			self.action_prob = q_network(self.state, self.action, state_dim, action_space, self.initializer)
			#self.action_prob_old = policy_nn_ad2(self.state, self.action, state_dim, action_space, self.initializer,trainable=True)
			self.action_prob_old = q_network_old(self.state, self.action, state_dim, action_space, self.initializer)
			self.crossE = tf.nn.softmax_cross_entropy_with_logits(labels = self.action_prob, logits =self.action_prob_old)
			self.accr_subj_test = tf.reduce_mean(-self.crossE)
			#self.kl = tf.distributions.kl_divergence(self.action_prob, self.action_prob_old)
			action_mask = tf.cast(tf.one_hot(self.action, depth = action_space), tf.bool)
			self.picked_action_prob = tf.boolean_mask(self.action_prob, action_mask)
			self.loss = tf.reduce_sum(-tf.log(self.picked_action_prob)+self.tflam*self.accr_subj_test) + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope = 'supervised_policy'))
			self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
			self.train_op = self.optimizer.minimize(self.loss)

	def predict(self, state, action, sess = None):
		sess = sess or tf.get_default_session()
		return sess.run(self.action_prob, {self.state: state, self.action:action})

	def update(self, state, action, alpha, sess = None):
		sess = sess or tf.get_default_session()
		_, loss, kl, ac, pac = sess.run([self.train_op, self.loss, self.accr_subj_test, self.action, self.picked_action_prob], {self.state: state, self.action: action, self.tflam:alpha})
		if kl < METHOD['kl_target'] / 1.1:  # adaptive lambda, this is in OpenAI's paper
			METHOD['lam'] /= 2
		elif kl > METHOD['kl_target'] * 1.1:
			METHOD['lam'] *= 2
		print(ac, ac.shape)
		print(pac, pac.shape)
		return loss

def multi_f(sample0, sample1, env):
    try:
        good_episodes_test = teacher(sample0, sample1, 5, env, graphpath)
    except Exception as e:
        print('Cannot find a path')
        good_episodes_test = 'None'
    return good_episodes_test

def handler(sample0,sample1,env):
    p = multiprocessing.Pool(processes=4)
    #r=p.map(multi_f, agent_num)
    r=p.starmap(multi_f,[(sample0,sample1,env)]*4)
    return r

def train():
	tf.reset_default_graph()
	policy_nn = SupervisedPolicy()
	f = open(relationPath)
	train_data = f.readlines()
	f.close()

	num_samples = len(train_data)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		if num_samples > 500:
			num_samples = 500
		else:
			num_episodes = num_samples

		pf = [0] * 200
		pcount = 0

		for episode in range(num_samples):
			print("Episode %d" % episode)
			print('Training Sample:', train_data[episode%num_samples][:-1])

			env = Env(dataPath, train_data[episode%num_samples])
			sample = train_data[episode%num_samples].split()

			#try:
			#	good_episodes = teacher(sample[0], sample[1], 5, env, graphpath)
			#	print(good_episodes.shape)
			#except Exception as e:
			#	print('Cannot find a path')
			#	continue
			good_episodes_test = multi_f(sample[0], sample[1], env)
			if good_episodes_test == 'None':
				continue

			#output_state_test.append(handler(sample[0],sample[1],env))
			for item in good_episodes_test:
				state_batch = []
				action_batch = []
				for t, transition in enumerate(item):
					state_batch.append(transition.state)
					action_batch.append(transition.action)
				state_batch = np.squeeze(state_batch)
				state_batch = np.reshape(state_batch, [-1, state_dim])
				pi = [0] * 200
				p = [0] * 200
				kl_p = 0
				for i in range(state_batch.shape[0]):
					p = state_batch[i]
					pi = pi +p
					pcount = pcount + 1
				pf = pf + pi
				cos_sim = dot(pf, pi) / (norm(pf) * norm(pi))
				#print('cos_sim',cos_sim)
				#a = open('test.txt', 'a+')
				#a.write(str(cos_sim)+'\n')
				policy_nn.update(state_batch, action_batch, METHOD['lam']*math.exp(1-cos_sim))
				#action_probs = policy_nn.predict(np.asarray([state_batch[0]]), np.asarray([action_batch[0]]))

		saver.save(sess, 'models/policy_supervised_' + relation)
		print('Model saved')
		#a.close()


def test(test_episodes):
	tf.reset_default_graph()
	policy_nn = SupervisedPolicy()

	f = open(relationPath)
	test_data = f.readlines()
	f.close()

	test_num = len(test_data)

	test_data = test_data[-test_episodes:]
	print(len(test_data))
	
	success = 0

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, 'models/policy_supervised_'+ relation)
		print('Model reloaded')
		for episode in range(len(test_data)):
			print('Test sample %d: %s' % (episode,test_data[episode][:-1]))
			env = Env(dataPath, test_data[episode])
			sample = test_data[episode].split()
			state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]
			good_episodes_test = multi_f(sample[0], sample[1], env)
			if good_episodes_test == 'None':
				continue

			# output_state_test.append(handler(sample[0],sample[1],env))
			for item in good_episodes_test:
				state_batch = []
				action_batch = []
				for t, transition in enumerate(item):
					state_batch.append(transition.state)
					action_batch.append(transition.action)
			action_idx = [env.relation2id_[sample[2]],0]
			for t in count():
				state_vec = env.idx_state(state_idx)
				#print([action_batch[0]].shape)
				action_probs = policy_nn.predict(state_vec,np.asarray([action_batch[0]]))

				action_chosen = np.random.choice(np.arange(action_space), p = np.squeeze(action_probs))
				reward, new_state, done = env.interact(state_idx, action_chosen)
				if done or t == max_steps_test:
					if done:
						print('Success')
						success += 1
					print('Episode ends\n')
					break
				state_idx = new_state

	print('Success persentage:', success/test_episodes)

if __name__ == "__main__":
	train()
	test(50)

