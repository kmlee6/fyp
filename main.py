import tensorflow as tf
from config import *
from utils import *
from loss import loss
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def nextBatch(data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

if __name__ == "__main__":
	#get style class, images, and images label
	style_dict = init_style_dict(data_path_+'**/')
	samples_path = get_images_path(data_path_, '*/*.jpg')
	samples = get_images(samples_path)
	samples_label = get_images_label(samples_path, style_dict)

	# define placeholder
	images = tf.placeholder(tf.float32, shape = [None,256,256,3], name='x_placeholder') #input images
	image_labels = tf.placeholder(tf.float32, shape = [None, 3], name='y_placeholder') #class labels
	noise = tf.placeholder(tf.float32, [None, 100], name='z_placeholder') #noise for generator

	d_opt, g_opt, sampler_, monitor, accuracy = loss(images, image_labels, noise)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	for i in range(len(samples_label)):
		z_batch = np.random.normal(0, 1, size=[batch_size, 100])
		image_batch, labels = nextBatch(samples, samples_label)
		print(labels)
		# print (image_batch.shape)
		# train discriminator	
		sess.run([d_opt, g_opt],{images: image_batch, image_labels: labels, noise: z_batch})
		z_batch = np.random.normal(0, 1, size=[batch_size, 100])
		# mon_0, mon_1 = sess.run([accuracy, monitor[1]], feed_dict={images: image_batch, image_labels: labels, noise: z_batch})
		if(i%10==0):
			sample_batch = np.random.normal(0, 1, size=[4, 100])
			output_ = sess.run(sampler_, feed_dict={noise: sample_batch})
			output_ = np.array(output_)
			save_images(output_, (2,2), './tmp_{}.jpg'.format(i))
		# print("Epoch {}: D-{}\n==\n{}".format(i, mon_0, mon_1))
