# import tensorflow as tensorflow
# import discriminator
# import generator
from utils import *
import neuralnet
import numpy as np

if __name__ == "__main__":
	#get style class, images, and images label
	style_dict = init_style_dict('./data/**/')
	samples_path = get_images_path('./data', '*/*.jpg')
	samples = get_images(samples_path)
	samples_label = get_images_label(samples_path, style_dict)

	#build model
	generator = neuralnet.generator
	discriminator = neuralnet.discriminator

	t_vars = tf.trainable_variables()
	d_vars = [var for var in t_vars if 'discriminator' in var.name]
	g_vars = [var for var in t_vars if 'generator' in var.name]

	#train model
	tf.global_variables_initializer().run()
	
