import tensorflow as tf

def value_nn(state, action, state_dim, action_space, initializer):
	w1 = tf.get_variable('w1', [201, 64], initializer = initializer)
	b1 = tf.get_variable('b1', [64], initializer = tf.constant_initializer(0.0))
	h1 = tf.nn.relu(tf.matmul(tf.concat([state, tf.expand_dims(tf.dtypes.cast(action, tf.float32),1)], 1),w1) + b1)
	w2 = tf.get_variable('w2', [64,1], initializer = initializer)
	b2 = tf.get_variable('b2', [1], initializer = tf.constant_initializer(0.0))
	value_estimated = tf.matmul(h1, w2) + b2
	return tf.squeeze(value_estimated)

def value_nn_old(state, action, state_dim, action_space, initializer):
	w1 = tf.get_variable('w1', [201, 64], initializer = initializer)
	b1 = tf.get_variable('b1', [64], initializer = tf.constant_initializer(0.0))
	h1 = tf.nn.relu(tf.matmul(tf.concat([state, tf.expand_dims(tf.dtypes.cast(action, tf.float32),1)], 1),w1) + b1)
	w2 = tf.get_variable('w2', [64,1], initializer = initializer)
	b2 = tf.get_variable('b2', [1], initializer = tf.constant_initializer(0.0))
	value_estimated = tf.stop_gradient(tf.matmul(h1, w2) + b2)
	return tf.squeeze(value_estimated)

def q_network(state, action, state_dim, action_space, initializer):
	w1 = tf.get_variable('w1', [201, 64], initializer=initializer)
	b1 = tf.get_variable('b1', [64], initializer = tf.constant_initializer(0))
	h3 = tf.nn.relu(tf.matmul( tf.concat([state, tf.expand_dims(tf.dtypes.cast(action, tf.float32),1)], 1), w1) + b1)
	w4 = tf.get_variable('w4', [64, 512], initializer = initializer)
	b4 = tf.get_variable('b4', [512], initializer = tf.constant_initializer(0))
	h4 = tf.nn.relu(tf.matmul(h3, w4) + b4)
	w5 = tf.get_variable('w5', [512, action_space], initializer = initializer)
	b5 = tf.get_variable('b5', [action_space], initializer = tf.constant_initializer(0))
	action_values = tf.matmul(h4, w5) + b5
	action_prob = tf.nn.softmax(tf.matmul(h4, w5) + b5)
	return action_prob