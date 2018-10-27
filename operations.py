def relu(x, slope=0.2):
	return tf.maximun(x, slope*x)

def conv2d(input_tensor, filter_shape, name):
	with tf.variable_scope(name):
		weights = tf.get_variable('w', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.02))
		bias = tf.get_variable('b', filter_shape[-1], initializer=tf.constant_initializer(0))
		output = tf.nn.conv2d(input=images, filter=weights, strides=[1, 2, 2, 1], padding='SAME')
		output = output + bias
		return output

def fully_connected(input_, input_len, output_len, name):
	with tf.variable_scope(name):
		weights = tf.get_variable('d_w7', [input_len, output_len], initializer=tf.truncated_normal_initializer(stddev=0.02))
		bias = tf.get_variable('d_b7', [output_len], initializer=tf.constant_initializer(0))
		output = tf.matmul(input_, weights) + bias
		return output

def fsconv2d(input_tensor, output_shape, name):
	with tf.variable_scope(name):
		weight_shape = (5, 5, output_shape.[-1], input_tensor.shape()[-1])
		weights = tf.get_variable('w', weight_shape, initializer=tf.random_normal_initializer(stddev=0.02))
		bias = tf.get_variable('b', output_shape[-1], initializer=tf.constant_initializer(0.0))
		output = tf.nn.conv2d_transpose(input_tensor, weights, output_shape=output_shape, strides=[1, 2, 2, 1])
		output = output + bias
		return output