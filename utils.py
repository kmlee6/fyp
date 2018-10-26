import os
from glob import glob
import scipy.misc
import numpy as np

def read_image(path):
	image = scipy.misc.imread(path).astype(np.float)
	cropped_image = scipy.misc.imresize(image, [256, 256])
	return np.array(cropped_image)/127.5 - 1. #normalization

############# the functions below are exported #############

def init_style_dict(path): #path = './data/**/'
	label_dict = {}
	style_folders = glob(path, recursive=True)[1:]
	for index, path in enumerate(style_folders):
		print(path[7:-1])
		label_dict[path[7:-1]] = index
	return label_dict

def get_images_path(path, filename_pattern): #dir = './data' pattern = '*/*.jpg'
	return glob(os.path.join(path, filename_pattern))

def get_images(sample_files):
	samples = [read_image(sample_file) for sample_file in sample_files]
	samples = np.array(samples).astype(np.float32)
	return samples

def get_images_label(images_path, label_dict):
	ret = []
	for path in images_path:
		_, _, label_str, _ = path.split('/', 3)
		ret.append(np.eye(3)[np.array(label_dict[label_str])])
	return ret