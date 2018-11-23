from operations import *
from config import *
import tensorflow as tf

def discriminator(images, reuse_variables = None):
	with tf.variable_scope("discriminator", reuse=reuse_variables) as scope:
		#conv 1 (32 4x4 filters)
		d1 = conv2d(images, [4, 4, 3, 32], 'd1')
		d1 = batch_norm(d1)
		d1 = tf.nn.leaky_relu(alpha=0.2, features = d1)

		#conv 2 (64 4x4 filters)
		d2 = conv2d(d1, [4, 4, 32, 64], 'd2')
		d2 = batch_norm(d2)
		d2 = tf.nn.leaky_relu(alpha=0.2, features =d2)

		#conv 3 (128 4x4 filters)
		d3 = conv2d(d2, [4, 4, 64, 128], 'd3')
		d3 = batch_norm(d3)
		d3 = tf.nn.leaky_relu(alpha=0.2, features =d3)

		#conv 4 (256 4x4 filters)
		d4 = conv2d(d3, [4, 4, 128, 256], 'd4')
		d4 = batch_norm(d4)
		d4 = tf.nn.leaky_relu(alpha=0.2, features =d4)

		#conv 5 (512 4x4 filters)
		d5 = conv2d(d4, [4, 4, 256, 512], 'd5')
		d5 = batch_norm(d5)
		d5 = tf.nn.leaky_relu(alpha=0.2, features =d5)

		#conv 6 (512 4x4 filters)
		d6 = conv2d(d5, [4, 4, 512, 512], 'd6')
		d6 = batch_norm(d6)
		d6 = tf.nn.leaky_relu(alpha=0.2, features =d6)

		d6 = tf.reshape(d6, [-1, 4 * 4 * 512])
		#fully connected layer to determine whether the image is real or fake
		d7 = fully_connected(d6, 4 * 4 * 512, 1, 'd7')
		d7_ = tf.nn.sigmoid(d7)

        #fully connect layer to classify the image into the different styles
        #first fully connected layer
		d8 = fully_connected(d6, 4 * 4 * 512, 1024, 'd8')
		d8 = tf.nn.leaky_relu(alpha=0.2, features =d8)

		#second fully connected layer
		d9 = fully_connected(d8, 1024, 512, 'd9')
		d9 = tf.nn.leaky_relu(alpha=0.2, features =d9)

		#third fully connected layer
		d10 = fully_connected(d9, 512, class_, 'd10')
		d10_ = tf.nn.softmax(d10)

		return d7, d7_, d10, d10_

def generator(batch_size, z, reuse_variables = None):
	with tf.variable_scope("generator", reuse=reuse_variables) as scope:
		g_w0 = tf.get_variable('g_w0', [100, 4*4*1024], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b0 = tf.get_variable('g_b0', [4*4*1024], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))

		#project and reshape
		g0 = tf.matmul(z, g_w0) + g_b0
		g0 = tf.reshape(g0, [batch_size, 4, 4, 1024])
		g0 = batch_norm(g0)
		g0 = tf.nn.relu(g0)

		#fsconv1
		g1 = fsconv2d(g0, [batch_size, 8, 8, 1024], 'g1')
		g1 = batch_norm(g1)
		g1 = relu(g1)

		#fsconv2
		g2 = fsconv2d(g1, [batch_size, 16, 16, 512], 'g2')
		g2 = batch_norm(g2)
		g2 = relu(g2)
		#fsconv3
		g3 = fsconv2d(g2, [batch_size, 32, 32, 256], 'g3')
		g3 = batch_norm(g3)
		g3 = relu(g3)
		#fsconv4
		g4 = fsconv2d(g3, [batch_size, 64, 64, 128], 'g4')
		g4 = batch_norm(g4)
		g4 = relu(g4)
		#fsconv5
		g5 = fsconv2d(g4, [batch_size, 128, 128, 64], 'g5')
		g5 = batch_norm(g5)
		g5 = relu(g5)
		#fsconv6
		g6 = fsconv2d(g5, [batch_size, 256, 256, 3], 'g6')
		g6 = batch_norm(g6)
		g6 = relu(g6)

		return tf.nn.tanh(g6)