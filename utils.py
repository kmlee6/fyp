# Version log:
# v1.0 initial version created.
# v1.1 removed depreciated scipy.imread, added skimage, imageio to read input


import os
import scipy
from glob import glob
import numpy as np
from skimage import transform
# import imageio


def transform_image(path):
	# removed in v1.1
	image = scipy.misc.imread(path).astype(np.float)
	cropped_image = scipy.misc.imresize(image, [256, 256])
	# image = imageio.imread(path, as_gray=False) 
	# cropped_image = transform.resize(image, (256,256), preserve_range=True, mode='constant')
	return np.array(cropped_image)/127.5 - 1 #normalization

def inverse_transform_image(data):
	return np.array(((data+1.) * 127.5 ) %255, dtype='uint8')

def merge_images(images, size):
	heigh, width = images.shape[1], images.shape[2]
	img = np.zeros((heigh * size[0], width * size[1], 3))
	print("haha")
	for index, image in enumerate(images):
		i = index% size[1]
		j = index // size[1]
		img[j * heigh:j * heigh + heigh, i * width:i * width + width, :] = image
	return img

############# the functions below are exported #############

def init_style_dict(path): #path = './data/**/'
	label_dict = {}
	style_folders = glob(path, recursive=True)[1:]
	for index, path in enumerate(style_folders):
		# print(path[7:-1])
		class_name = path.split('/')[-2]
		label_dict[class_name] = index
	print('{{Style dictionary:}}')
	print(label_dict)
	return label_dict

def get_images_path(path, filename_pattern): #dir = './data' pattern = '*/*.jpg'
	return glob(os.path.join(path, filename_pattern))

def get_images(sample_files):
	samples = [transform_image(sample_file) for sample_file in sample_files]
	samples = np.array(samples).astype(np.float32)
	return samples

def get_images_label(images_path, label_dict):
	ret = []
	for path in images_path:
		_, _, label_str, _ = path.split('/', 3)
		ret.append(np.eye(3)[np.array(label_dict[label_str])])
	return ret

def save_images(images, size, images_path):
	converted_ = inverse_transform_image(images)
	# merged_ = merge_images(converted_, size)
	image = np.squeeze(merge_images(converted_, size))
	# imageio.imwrite(images_path, image)
	scipy.misc.imsave(images_path, image)
