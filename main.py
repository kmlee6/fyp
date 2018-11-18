import tensorflow as tf
from utils import *
from loss import loss
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
	images = tf.placeholder(tf.float32, shape = [None,256,256,3], name='x_placeholder') #input images
	image_labels = tf.placeholder(tf.float32, shape = [None, 3], name='y_placeholder') #class labels
	noise = tf.placeholder(tf.float32, [None, 100], name='z_placeholder') #noise for generator

	d_opt, g_opt, monitor, accuracy = loss(batch_size, images, image_labels, noise)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	for i in range(len(samples_label)):
		z_batch = np.random.normal(0, 1, size=[batch_size, 100])
		image_batch, labels = nextBatch(batch_size, samples, samples_label)
		# print (image_batch.shape)
		# train discriminator	
		sess.run([d_opt, g_opt],{images: image_batch, image_labels: labels, noise: z_batch})
		# train generator
		z_batch = np.random.normal(0, 1, size=[batch_size, 100])
		d_loss, g_loss = sess.run([monitor[2], monitor[3]], feed_dict={images: image_batch, image_labels: labels, noise: z_batch})
		print("Epoch {}: D-{} G-{}".format(i, d_loss, g_loss))
