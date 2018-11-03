import tensorflow as tf
from utils import *
import neuralnet
import numpy as np

# parameters:
batch_size = 10

def nextBatch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

if __name__ == "__main__":
	#get style class, images, and images label
	style_dict = init_style_dict('./data/**/')
	samples_path = get_images_path('./data', '*/*.jpg')
	samples = get_images(samples_path)
	samples_label = get_images_label(samples_path, style_dict)

	# define placeholder
	x_placeholder = tf.placeholder(tf.float32, shape = [None,256,256,3], name='x_placeholder')
	z_placeholder = tf.placeholder(tf.float32, [None, 100], name='z_placeholder')

	# define input for each model
	generator = neuralnet.generator(z_placeholder)
	discriminator_real, class_ = neuralnet.discriminator(x_placeholder)
	discriminator_fake, class_ = neuralnet.discriminator(generator, reuse_variables = True)

	# define loss functions
	discriminator_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = discriminator_real, labels = tf.ones_like(discriminator_real)))
	discriminator_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = discriminator_fake, labels = tf.zeros_like(discriminator_fake)))
	generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = discriminator_fake, labels = tf.ones_like(discriminator_fake)))
	
	# initialize the variables for the models	
	t_vars = tf.trainable_variables()
	d_vars = [var for var in t_vars if 'discriminator' in var.name]
	g_vars = [var for var in t_vars if 'generator' in var.name]

	Optimize_function_fake = tf.train.AdamOptimizer(0.0003).minimize(discriminator_loss_fake, var_list=d_vars)
	Optimize_function_real = tf.train.AdamOptimizer(0.0003).minimize(discriminator_loss_real, var_list=d_vars)

	tf.get_variable_scope().reuse_variables()
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	for i in range(len(samples_label)):
		z_batch = np.random.normal(0, 1, size=[batch_size, 100])
		image_batch, x = nextBatch(batch_size, samples, samples_label)
		# print (image_batch.shape)
		# train discriminator	
		sess.run([Optimize_function_real, Optimize_function_fake],{x_placeholder: image_batch, z_placeholder: z_batch})

		# train generator
		z_batch = np.random.normal(0, 1, size=[batch_size, 100])
		_ = sess.run(generator, feed_dict={z_placeholder: z_batch})
	
	#train model
	# tf.global_variables_initializer().run()
	# sess.run([d_trainer_real, d_trainer_fake],{x_placeholder: real_image_batch, z_placeholder: z_batch})
	
