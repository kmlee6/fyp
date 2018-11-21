import neuralnet
import tensorflow as tf
from config import *

def loss(images, image_labels, noise):
	# define input for each model
	g_ = neuralnet.generator(batch_size, noise)
	sampler_ = neuralnet.generator(4, noise, reuse_variables = True)
	d_real, d_real_c = neuralnet.discriminator(images)
	d_fake, d_fake_c = neuralnet.discriminator(g_, reuse_variables = True)

	correct_prediction = tf.equal(tf.argmax(image_labels,1), tf.argmax(d_real_c,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	#define loss functions
	real_detect = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_real, labels = tf.ones_like(d_real)))
	d_fake_detect = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake, labels = tf.zeros_like(d_fake)))

	real_c = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_real_c, labels = image_labels))
	fake_c = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake_c, labels = (1.0/ class_) * tf.ones_like(d_fake_c)))# 1 / num of class

	g_fake_detect = -tf.reduce_mean(tf.log(d_fake))

	d_loss = real_detect + real_c + d_fake_detect
	g_loss = g_fake_detect + fake_c

	#let optimizer
	t_vars = tf.trainable_variables()
	d_vars = [var for var in t_vars if 'discriminator' in var.name]
	g_vars = [var for var in t_vars if 'generator' in var.name]

	d_opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5).minimize(d_loss, var_list=d_vars)
	g_opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5).minimize(g_loss, var_list=g_vars)

	tf.get_variable_scope().reuse_variables()

	return d_opt, g_opt, sampler_, [d_real, d_fake], correct_prediction