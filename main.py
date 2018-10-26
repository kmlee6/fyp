# import tensorflow as tensorflow
# import discriminator
# import generator
from utils import *
import numpy as np

if __name__ == "__main__":
	style_dict = init_style_dict('./data/**/')
	samples_path = get_images_path('./data', '*/*.jpg')
	samples = get_images(samples_path)
	samples_label = get_images_label(samples_path, style_dict)
	for i in samples_label:
		print(i)