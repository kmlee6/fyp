import neuralnet
import tensorflow as tf
from config import *

def loss(images, image_labels, noise):
	# define input for each model
	g_ = neuralnet.generator(batch_size, noise)
	sampler_ = neuralnet.generator(4, noise, reuse_variables = True)
	d_real_logits, d_real, d_real_c_logits, d_real_c = neuralnet.discriminator(images)

	# g_with_noise = g_ + tf.random_normal(shape=tf.shape(g_), mean=0.0, stddev=0.1)
	d_fake_logits, d_fake, d_fake_c_logits, d_fake_c = neuralnet.discriminator(g_, reuse_variables = True)

	true_label = tf.random_uniform(tf.shape(d_real),.8, 1.2)
	false_label = tf.random_uniform(tf.shape(d_fake), 0.0, 0.3)

	#define loss functions
	real_detect = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_real_logits, labels = true_label*tf.ones_like(d_real)))
	d_fake_detect = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake_logits, labels = false_label*tf.ones_like(d_fake)))

	real_c = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_real_c_logits, labels = image_labels))
	fake_c = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake_c_logits, labels = (1.0/ class_) * tf.ones_like(d_fake_c)))# 1 / num of class

	g_fake_detect = -tf.reduce_mean(tf.log(d_fake))

	d_loss = real_detect + real_c + d_fake_detect
	g_loss = g_fake_detect + fake_c

	correct_prediction = tf.equal(tf.argmax(image_labels,1), tf.argmax(d_real_c,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	real_prob = tf.reduce_mean(d_real)
	fake_prob = tf.reduce_mean(d_fake)
	tf.summary.scalar('real_probability', real_prob)
	tf.summary.scalar('fake_probability', fake_prob)
	tf.summary.scalar('real_classify', accuracy)

	tf.summary.scalar('d_loss_real_detect', real_detect)
	tf.summary.scalar('d_loss_fake_detect', d_fake_detect)
	tf.summary.scalar('d_loss_real_class', real_c)
	tf.summary.scalar('g_loss_fake_detect', g_fake_detect)
	tf.summary.scalar('g_loss_fake_class', fake_c)

	#let optimizer
	t_vars = tf.trainable_variables()
	d_vars = [var for var in t_vars if 'discriminator' in var.name]
	g_vars = [var for var in t_vars if 'generator' in var.name]

	d_opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.4).minimize(d_loss, var_list=d_vars)
	g_opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.4).minimize(g_loss, var_list=g_vars)

	tf.get_variable_scope().reuse_variables()

	merged = tf.summary.merge_all()

	return d_opt, g_opt, sampler_, [accuracy, d_fake], merged