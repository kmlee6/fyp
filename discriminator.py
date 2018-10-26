import tensorflow as tf

def discriminator(images, reuse_variables = None):
	with tf.variable_scope("discriminator", reuse=reuse_variables) as scope:
		#conv 1 (32 4x4 filters)
		d_w1 = tf.get_variable('d_w1', [4, 4, 3, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
		d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 2, 2, 1], padding='SAME')
		d1 = d1 + d_b1
		d1 = tf.nn.leaky_relu(alpha=0.2, d1)

		#conv 2 (64 4x4 filters)
		d_w2 = tf.get_variable('d_w2', [4, 4, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
		d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 2, 2, 1], padding='SAME')
		d2 = d2 + d_b2
		d2 = tf.nn.leaky_relu(alpha=0.2, d2)

		#conv 3 (128 4x4 filters)
		d_w3 = tf.get_variable('d_w3', [4, 4, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b3 = tf.get_variable('d_b3', [128], initializer=tf.constant_initializer(0))
		d3 = tf.nn.conv2d(input=d2, filter=d_w3, strides=[1, 2, 2, 1], padding='SAME')
		d3 = d3 + d_b3
		d3 = tf.nn.leaky_relu(alpha=0.2, d3)

		#conv 4 (256 4x4 filters)
		d_w4 = tf.get_variable('d_w1', [4, 4, 64, 126], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b4 = tf.get_variable('d_b1', [256], initializer=tf.constant_initializer(0))
		d4 = tf.nn.conv2d(input=d3, filter=d_w4, strides=[1, 2, 2, 1], padding='SAME')
		d4 = d4 + d_b4
		d4 = tf.nn.leaky_relu(alpha=0.2, d4)

		#conv 5 (512 4x4 filters)
		d_w5 = tf.get_variable('d_w5', [4, 4, 256, 512], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b5 = tf.get_variable('d_b5', [512], initializer=tf.constant_initializer(0))
		d5 = tf.nn.conv2d(input=d4, filter=d_w5, strides=[1, 2, 2, 1], padding='SAME')
		d5 = d5 + d_b5
		d5 = tf.nn.leaky_relu(alpha=0.2, d5)

		#conv 6 (512 4x4 filters)
		d_w6 = tf.get_variable('d_w6', [4, 4, 512, 512], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b6 = tf.get_variable('d_b6', [512], initializer=tf.constant_initializer(0))
		d6 = tf.nn.conv2d(input=d5, filter=d_w6, strides=[1, 2, 2, 1], padding='SAME')
		d6 = d6 + d_b6
		d6 = tf.nn.leaky_relu(alpha=0.2, d6)

		#fully connected layer to determine whether the image is real or fake
		d_w7 = tf.get_variable('d_w7', [4 * 4 *512, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b7 = tf.get_variable('d_b7', [1], initializer=tf.constant_initializer(0))
		d7 = tf.matmul(d6, d_w7) + d_b7
		d7 = tf.nn.sigmoid(d7)

        #fully connect layer to classify the image into the different styles
        #first fully connected layer
		d_w8 = tf.get_variable('d_w8', [4 * 4 *512, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b8 = tf.get_variable('d_b8', [1024], initializer=tf.constant_initializer(0))
		d8 = tf.matmul(d6, d_w8) + d_b8
		d8 = tf.nn.leaky_relu(alpha=0.2, d8)

		#second fully connected layer
		d_w9 = tf.get_variable('d_w9', [1024, 512], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b9 = tf.get_variable('d_b9', [512], initializer=tf.constant_initializer(0))
		d9 = tf.matmul(d8, d_w9) + d_b9
		d9 = tf.nn.leaky_relu(alpha=0.2, d9)

		#third fully connected layer
		d_w10 = tf.get_variable('d_w10', [512, 3], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b10 = tf.get_variable('d_b10', [3], initializer=tf.constant_initializer(0))
		d10 = tf.matmul(d9, d_w10) + d_b10
		d10 = tf.nn.sigmoid(d10)

		return d7, d10
