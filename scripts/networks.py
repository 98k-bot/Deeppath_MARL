import tensorflow as tf 

def policy_nn(state, state_dim, action_dim, initializer,trainable):
	w1 = tf.get_variable('W1', [state_dim, 512], initializer = initializer, regularizer=tf.contrib.layers.l2_regularizer(0.01),trainable=trainable)
	b1 = tf.get_variable('b1', [512], initializer = tf.constant_initializer(0.0),trainable=trainable)
	h1 = tf.nn.relu(tf.matmul(state, w1) + b1)
	w2 = tf.get_variable('w2', [512, 1024], initializer = initializer, regularizer=tf.contrib.layers.l2_regularizer(0.01),trainable=trainable)
	b2 = tf.get_variable('b2', [1024], initializer = tf.constant_initializer(0.0),trainable=trainable)
	h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
	w3 = tf.get_variable('w3', [1024, action_dim], initializer = initializer, regularizer=tf.contrib.layers.l2_regularizer(0.01),trainable=trainable)
	b3 = tf.get_variable('b3', [action_dim], initializer = tf.constant_initializer(0.0),trainable=trainable)
	action_prob = tf.nn.softmax(tf.matmul(h2,w3) + b3)
	return action_prob

def policy_nn_ad(state, action, state_dim, action_dim, initializer,trainable):
	w1 = tf.get_variable('W1', [201, 512], initializer = initializer, regularizer=tf.contrib.layers.l2_regularizer(0.01),trainable=trainable)
	b1 = tf.get_variable('b1', [512], initializer = tf.constant_initializer(0.0),trainable=trainable)
	h1 = tf.nn.relu(tf.matmul(tf.concat([state, tf.expand_dims(tf.dtypes.cast(action, tf.float32),1)], 1), w1) + b1)
	w2 = tf.get_variable('w2', [512, 1024], initializer = initializer, regularizer=tf.contrib.layers.l2_regularizer(0.01),trainable=trainable)
	b2 = tf.get_variable('b2', [1024], initializer = tf.constant_initializer(0.0),trainable=trainable)
	h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
	w3 = tf.get_variable('w3', [1024, action_dim], initializer = initializer, regularizer=tf.contrib.layers.l2_regularizer(0.01),trainable=trainable)
	b3 = tf.get_variable('b3', [action_dim], initializer = tf.constant_initializer(0.0),trainable=trainable)
	action_prob = tf.nn.softmax(tf.matmul(h2,w3) + b3)
	return action_prob

def policy_nn2(state, state_dim, action_dim, initializer,trainable):
	w1 = tf.get_variable('W1', [state_dim, 512], initializer = initializer, regularizer=tf.contrib.layers.l2_regularizer(0.01),trainable=trainable)
	b1 = tf.get_variable('b1', [512], initializer = tf.constant_initializer(0.0),trainable=trainable)
	h1 = tf.stop_gradient(tf.nn.relu(tf.matmul(state, w1) + b1))
	w2 = tf.get_variable('w2', [512, 1024], initializer = initializer, regularizer=tf.contrib.layers.l2_regularizer(0.01),trainable=trainable)
	b2 = tf.get_variable('b2', [1024], initializer = tf.constant_initializer(0.0),trainable=trainable)
	h2 = tf.stop_gradient(tf.nn.relu(tf.matmul(h1, w2) + b2))
	w3 = tf.get_variable('w3', [1024, action_dim], initializer = initializer, regularizer=tf.contrib.layers.l2_regularizer(0.01),trainable=trainable)
	b3 = tf.get_variable('b3', [action_dim], initializer = tf.constant_initializer(0.0),trainable=trainable)
	action_prob = tf.stop_gradient(tf.nn.softmax(tf.matmul(h2,w3) + b3))
	return action_prob

def policy_nn_ad2(state, action, state_dim, action_dim, initializer,trainable):
	w1 = tf.get_variable('W1', [201, 512], initializer = initializer, regularizer=tf.contrib.layers.l2_regularizer(0.01),trainable=trainable)
	b1 = tf.get_variable('b1', [512], initializer = tf.constant_initializer(0.0),trainable=trainable)
	h1 = tf.nn.relu(tf.matmul(tf.concat([state, tf.expand_dims(tf.dtypes.cast(action, tf.float32),1)], 1), w1) + b1)
	w2 = tf.get_variable('w2', [512, 1024], initializer = initializer, regularizer=tf.contrib.layers.l2_regularizer(0.01),trainable=trainable)
	b2 = tf.get_variable('b2', [1024], initializer = tf.constant_initializer(0.0),trainable=trainable)
	h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
	w3 = tf.get_variable('w3', [1024, action_dim], initializer = initializer, regularizer=tf.contrib.layers.l2_regularizer(0.01),trainable=trainable)
	b3 = tf.get_variable('b3', [action_dim], initializer = tf.constant_initializer(0.0),trainable=trainable)
	action_prob = tf.stop_gradient(tf.nn.softmax(tf.matmul(h2,w3) + b3))
	return action_prob

def value_nn(state, action, state_dim, action_space, initializer):
	w4 = tf.get_variable('w4', [201, 64], initializer = initializer)
	b4 = tf.get_variable('b4', [64], initializer = tf.constant_initializer(0.0))
	h4 = tf.nn.relu(tf.matmul(tf.concat([state, tf.expand_dims(tf.dtypes.cast(action, tf.float32),1)], 1),w4) + b4)
	w5 = tf.get_variable('w5', [64,1], initializer = initializer)
	b5 = tf.get_variable('b5', [1], initializer = tf.constant_initializer(0.0))
	value_estimated = tf.matmul(h4, w5) + b5
	return tf.squeeze(value_estimated)

def value_nn_old(state, action, state_dim, action_space, initializer):
	w4 = tf.get_variable('w4', [201, 64], initializer = initializer)
	b4 = tf.get_variable('b4', [64], initializer = tf.constant_initializer(0.0))
	h4 = tf.nn.relu(tf.matmul(tf.concat([state, tf.expand_dims(tf.dtypes.cast(action, tf.float32),1)], 1),w4) + b4)
	w5 = tf.get_variable('w5', [64,1], initializer = initializer)
	b5 = tf.get_variable('b5', [1], initializer = tf.constant_initializer(0.0))
	value_estimated = tf.stop_gradient(tf.matmul(h4, w5) + b5)
	return tf.squeeze(value_estimated)

def q_network(state, action, state_dim, action_space, initializer):
	w1 = tf.get_variable('w1', [201, 512], initializer=initializer)
	b1 = tf.get_variable('b1', [512], initializer = tf.constant_initializer(0))
	h1 = tf.nn.relu(tf.matmul( tf.concat([state, tf.expand_dims(tf.dtypes.cast(action, tf.float32),1)], 1), w1) + b1)
	w2 = tf.get_variable('w2', [512, 1024], initializer = initializer)
	b2 = tf.get_variable('b2', [1024], initializer = tf.constant_initializer(0))
	h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
	w3 = tf.get_variable('w3', [1024, action_space], initializer = initializer)
	b3 = tf.get_variable('b3', [action_space], initializer = tf.constant_initializer(0))
	action_values = tf.matmul(h2, w3) + b3
	action_prob = tf.nn.softmax(tf.matmul(h2, w3) + b3)
	return action_prob

def q_network_old(state, action, state_dim, action_space, initializer):
	w1 = tf.get_variable('w1', [201, 512], initializer=initializer)
	b1 = tf.get_variable('b1', [512], initializer = tf.constant_initializer(0))
	h1 = tf.nn.relu(tf.matmul( tf.concat([state, tf.expand_dims(tf.dtypes.cast(action, tf.float32),1)], 1), w1) + b1)
	w2 = tf.get_variable('w2', [512, 1024], initializer = initializer)
	b2 = tf.get_variable('b2', [1024], initializer = tf.constant_initializer(0))
	h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
	w3 = tf.get_variable('w3', [1024, action_space], initializer = initializer)
	b3 = tf.get_variable('b3', [action_space], initializer = tf.constant_initializer(0))
	action_values = tf.matmul(h2, w3) + b3
	action_prob = tf.stop_gradient(tf.nn.softmax(tf.matmul(h2, w3) + b3))
	return action_prob