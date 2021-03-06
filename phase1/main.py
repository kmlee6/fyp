from __future__ import division, print_function, absolute_import
from subprocess import call
import os

import tensorflow as tf
from config import *
from utils import *
from loss import loss
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def shuffle_data(data_path):
	idx = np.arange(0 , len(data_path))
	np.random.shuffle(idx)
	data_shuffled = [data_path[i] for i in idx]
	return data_shuffled

# def nextPool(data_path, labels, iter_):
# 	index = (iter_ // 10 )%3
# 	print("current pool index is {}".format(index))
# 	print("pool size is {}".format(pool_size))
# 	data_path_cluster = data_path[pool_size*index:pool_size*(index+1)]
# 	labels_cluster = labels[pool_size*index:pool_size*(index+1)]
# 	samples = get_images(data_path_cluster)
# 	print(len(data_path))
# 	return samples, labels_cluster
# 
# def nextBatch(data, labels):
# 	idx = np.arange(0 , len(data))
# 	np.random.shuffle(idx)
# 	idx = idx[:batch_size]
# 	data_shuffle = [data[i] for i in idx]
# 	labels_shuffle = [labels[i] for i in idx]
# 	return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def nextBatch(data_path, labels):
	idx = np.arange(0 , len(data_path))
	np.random.shuffle(idx)
	idx = idx[:batch_size]
	data_path_shuffle = [data_path[i] for i in idx]
	labels_shuffle = [labels[i] for i in idx]
	data = get_images(data_path_shuffle)
	return np.asarray(data), np.asarray(labels_shuffle)

if __name__ == "__main__":
	# mode = input("Which mode you want?")
	mode = "1"
	if mode == "1":
		print("-------------- GPU Usage ---------------")
		result = call(["nvidia-smi", "--format=csv", "--query-gpu=index,utilization.gpu,utilization.memory,memory.free"])
		if result == 0:
			chosen = input("Please choose one from the above (default as 0): ")
			# chosen = '1,2,3,5'
			if chosen == '':
				chosen = "0"
			os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen)
			config = tf.ConfigProto()
			config.gpu_options.per_process_gpu_memory_fraction = mem_fraction
			sess = tf.Session(config=config)
		else:
			exit(0)
	else:
		sess = tf.Session()

	#get style class, images, and images label
	style_dict = init_style_dict(data_path+'**/')
	ori_path = get_images_path(data_path, '*/*.jpg')
	samples_path = shuffle_data(ori_path)
	samples_label = get_images_label(samples_path, style_dict)

	# define placeholder
	images = tf.placeholder(tf.float32, shape = [None,256,256,3], name='x_placeholder') #input images
	image_labels = tf.placeholder(tf.float32, shape = [None, class_], name='y_placeholder') #class labels
	noise = tf.placeholder(tf.float32, [None, 100], name='z_placeholder') #noise for generator

	d_opt, g_opt, sampler_, monitor, merged = loss(images, image_labels, noise)
	saver = tf.train.Saver()

	if check_point == -1:
		sess.run(tf.global_variables_initializer())
	else:
		saver.restore(sess, model_path+"model_{}.ckpt".format(check_point))

	train_writer = tf.summary.FileWriter(log_path, sess.graph)

	for i in range(check_point+1, step_size):
		# if i%10 == 0:
		# 	pool_sample, pool_label = nextPool(samples_path, samples_label, i)
		# 	print("there are {} samples.".format(len(next_pool)))

		z_batch = np.random.normal(0, 1, size=[batch_size, 100])
		# image_batch, labels = nextBatch(pool_sample, pool_label)
		image_batch, labels = nextBatch(samples_path, samples_label)
		print("Iteration {}".format(i))
		# print(labels)
		# train discriminator	
		sess.run([d_opt, g_opt],{images: image_batch, image_labels: labels, noise: z_batch})

		# z_batch = np.random.normal(0, 1, size=[batch_size, 100])
		# mon_= sess.run([g_opt], feed_dict={noise: z_batch})
		if(i%10==0):
			summary_ = sess.run(merged, feed_dict={images: image_batch, image_labels: labels, noise: z_batch})
			train_writer.add_summary(summary_, i)
		if(i%100==0):
			z_batch = np.random.normal(0, 1, size=[batch_size, 100])
			mon_0, mon_1 = sess.run([monitor[0], monitor[1]], feed_dict={images: image_batch, image_labels: labels, noise: z_batch})
			print(mon_0)
			print("**************")
			print(mon_1)
			sample_batch = np.random.normal(0, 1, size=[4, 100])
			output_ = sess.run(sampler_, feed_dict={noise: sample_batch})
			output_ = np.array(output_)
			save_images(output_, (2,2), '{}/sample_{}.jpg'.format(experiment_path, i))
		if(i%100==0):
			save_path = saver.save(sess, "{}/model_{}.ckpt".format(model_path, i))
