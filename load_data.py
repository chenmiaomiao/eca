
import os

from scipy.io import loadmat
import numpy as np
from tqdm import tqdm, trange
import cv2

from sklearn.model_selection import train_test_split

import keras
from keras.preprocessing import sequence
from keras.datasets import mnist, imdb, cifar10
from keras import backend as K


# return: state_len, class_size, x_train, y_train, x_test, y_test 

def load_mnist():
	num_classes = 10
	# input image dimensions
	img_rows, img_cols = 28, 28
	state_len = img_rows * img_cols

	# the data, split between train and test sets
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	if K.image_data_format() == 'channels_first':
	    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	    input_shape = (img_rows, img_cols, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	x_train = x_train.reshape(-1, 1, state_len)
	x_test = x_test.reshape(-1, 1, state_len)
	print(np.dot(x_train[0], x_train[0].T))
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return state_len, num_classes, x_train, y_train, x_test, y_test

def load_2d():
	state_len = 2
	num_classes = 2
	# the data, split between train and test sets
	cls1 = np.load("art_data/data-2d-clsa.npy")
	cls2 = np.load("art_data/data-2d-clsb.npy")
	print(cls1.shape)
	#cls3 = cls3[:1000]

	#cls1[:, 2:] *= 10

	y_c1 = np.zeros((cls1.shape[0],))
	y_c2 = np.ones((cls2.shape[0],))
	print(y_c1.shape)
	print(y_c2.shape)

	x = np.concatenate([cls1, cls2], axis=0)
	y = np.concatenate([y_c1, y_c2], axis=0)

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=723)

	print(x_train.shape)
	print(x_test.shape)
	print(y_train.shape)
	print(y_test.shape)


	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= np.max(x_train)
	x_test /= np.max(x_train)
	x_train = x_train.reshape(-1, 1, state_len)
	x_test = x_test.reshape(-1, 1, state_len)

	print(np.dot(x_train[0], x_train[0].T))

	print('x_train shape:', x_train.shape)
	print('y_train shape:', y_train.shape)

	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return 2, 2, x_train, y_train, x_test, y_test

def load_3d():
	state_len = 3
	num_classes = 2
	# the data, split between train and test sets
	#(x_train, y_train), (x_test, y_test) = mnist.load_data()
	clsa = np.load("art_data/data-3d-clsa.npy")
	clsb = np.load("art_data/data-3d-clsb.npy")
	clsc = np.load("art_data/data-3d-clsc.npy")

	cls1 = clsa
	cls2 = np.concatenate([clsb, clsc], axis=0)
	#cls2 = np.concatenate([clsb[:2500], clsc[:2500]], axis=0)

	print(cls1.shape)
	print(cls2.shape)

	#cls1[:, 2:] *= 10

	y_c1 = np.zeros((cls1.shape[0],))
	y_c2 = np.ones((cls2.shape[0],))
	print(y_c1.shape)

	x = np.concatenate([cls1, cls2], axis=0)
	y = np.concatenate([y_c1, y_c2], axis=0)

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=723)

	print(x_train.shape)
	print(x_test.shape)
	print(y_train.shape)
	print(y_test.shape)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= np.max(x_train)
	x_test /= np.max(x_train)

	#x_train = wrap_norm(x_train, np.linalg.norm(x_train, axis=1, keepdims=True))
	#x_test = wrap_norm(x_test, np.linalg.norm(x_test, axis=1, keepdims=True))
	#state_len += 1

	x_train = x_train.reshape(-1, 1, state_len)
	x_test = x_test.reshape(-1, 1, state_len)

	print(np.dot(x_train[0], x_train[0].T))

	print('x_train shape:', x_train.shape)
	print('y_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')


	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return 3, 2, x_train, y_train, x_test, y_test

def load_imdb():
	num_classes = 2
	# set parameters:
	max_features = 500
	maxlen = 10

	imdb_magic_code = f"mf-{max_features}-ml-{maxlen}"

	# x_train_filename = f"/home/sm/Desktop/imdb_{imdb_magic_code}_x_train.npy"
	# x_test_filename = f"/home/sm/Desktop/imdb_{imdb_magic_code}_x_test.npy"
	# y_train_filename = f"/home/sm/Desktop/imdb_{imdb_magic_code}_y_train.npy"
	# y_test_filename = f"/home/sm/Desktop/imdb_{imdb_magic_code}_y_test.npy"
	# info_filename = f"/home/sm/Desktop/imdb_{imdb_magic_code}_info.npy"

	# data_save_root = "dataset/imdb"
	data_save_root = "/home/sm/Desktop/imdb"
	os.makedirs(data_save_root, exist_ok=True)

	x_train_filename = f"imdb_{imdb_magic_code}_x_train.npy"
	x_test_filename = f"imdb_{imdb_magic_code}_x_test.npy"
	y_train_filename = f"imdb_{imdb_magic_code}_y_train.npy"
	y_test_filename = f"imdb_{imdb_magic_code}_y_test.npy"
	info_filename = f"imdb_{imdb_magic_code}_info.npy"

	x_train_save_path = os.path.join(data_save_root, x_train_filename)
	x_test_save_path = os.path.join(data_save_root, x_test_filename)
	y_train_save_path = os.path.join(data_save_root, y_train_filename)
	y_test_save_path = os.path.join(data_save_root, y_test_filename)
	info_save_path = os.path.join(data_save_root, info_filename)

	if os.path.isfile(x_train_save_path) and \
		os.path.isfile(x_train_save_path) and \
		os.path.isfile(x_train_save_path) and \
		os.path.isfile(x_train_save_path):

		print("File exists.")

		x_train = np.load(x_train_save_path)
		x_test = np.load(x_test_save_path)
		y_train = np.load(y_train_save_path)
		y_test = np.load(y_test_save_path)
		state_len, num_classes = np.load(info_save_path)

		print("File loaded.")

		return state_len, num_classes, x_train, y_train, x_test, y_test

	print('Loading data...')
	(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
	print(len(x_train), 'train sequences')
	print(len(x_test), 'test sequences')

	print('Pad sequences (samples x time)')
	x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
	x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

	print('x_train shape:', x_train.shape)
	print('y_train shape:', y_train.shape)
	print('x_test shape:', x_test.shape)
	print('y_test shape:', y_test.shape)

	train_max = np.max(x_train)
	test_max =  np.max(x_test)
	voc_num = (train_max, test_max)[test_max > train_max] + 1
	state_len = maxlen * voc_num

	#x_train = np.stack([keras.utils.to_categorical(x, voc_num).reshape(-1) for x in x_train], axis=0)
	#x_test = np.stack([keras.utils.to_categorical(x, voc_num).reshape(-1) for x in x_test], axis=0)

	x_train_ = np.zeros((x_train.shape[0], state_len))
	x_test_ = np.zeros((x_test.shape[0], state_len))

	print("Expand dimension of x_train...")
	i = 0
	for x in tqdm(x_train):
		x = keras.utils.to_categorical(x, voc_num).reshape(-1)
		x_train_[i, :] = x
		i += 1

	print("Expand dimension of x_test...")
	i = 0
	for x in tqdm(x_test):
		x = keras.utils.to_categorical(x, voc_num).reshape(-1)
		x_test_[i, :] = x
		i += 1



	x_train = x_train_.reshape(-1, 1, state_len)
	x_test = x_test_.reshape(-1, 1, state_len)


	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)	

	print('x_train shape:', x_train.shape)
	print('y_train shape:', y_train.shape)
	print('x_test shape:', x_test.shape)
	print('y_test shape:', y_test.shape)


	np.save(x_train_save_path, x_train)
	np.save(x_test_save_path, x_test)
	np.save(y_train_save_path, y_train)
	np.save(y_test_save_path, y_test)
	np.save(info_save_path, np.array([state_len, num_classes]))

	return state_len, num_classes, x_train, y_train, x_test, y_test

def load_imdb_noembed():
	num_classes = 2
	# set parameters:
	max_features = 5000
	maxlen = 5000
	state_len = maxlen

	imdb_magic_code = f"noembed_mf-{max_features}-ml-{maxlen}"

	# x_train_filename = f"/home/sm/Desktop/imdb_{imdb_magic_code}_x_train.npy"
	# x_test_filename = f"/home/sm/Desktop/imdb_{imdb_magic_code}_x_test.npy"
	# y_train_filename = f"/home/sm/Desktop/imdb_{imdb_magic_code}_y_train.npy"
	# y_test_filename = f"/home/sm/Desktop/imdb_{imdb_magic_code}_y_test.npy"
	# info_filename = f"/home/sm/Desktop/imdb_{imdb_magic_code}_info.npy"

	# data_save_root = "dataset/imdb"
	data_save_root = "/home/sm/Desktop/imdb" 
	os.makedirs(data_save_root, exist_ok=True)

	x_train_filename = f"imdb_{imdb_magic_code}_x_train.npy"
	x_test_filename = f"imdb_{imdb_magic_code}_x_test.npy"
	y_train_filename = f"imdb_{imdb_magic_code}_y_train.npy"
	y_test_filename = f"imdb_{imdb_magic_code}_y_test.npy"
	info_filename = f"imdb_{imdb_magic_code}_info.npy"

	x_train_save_path = os.path.join(data_save_root, x_train_filename)
	x_test_save_path = os.path.join(data_save_root, x_test_filename)
	y_train_save_path = os.path.join(data_save_root, y_train_filename)
	y_test_save_path = os.path.join(data_save_root, y_test_filename)
	info_save_path = os.path.join(data_save_root, info_filename)

	if os.path.isfile(x_train_save_path) and \
		os.path.isfile(x_train_save_path) and \
		os.path.isfile(x_train_save_path) and \
		os.path.isfile(x_train_save_path):

		print("File exists.")

		x_train = np.load(x_train_save_path)
		x_test = np.load(x_test_save_path)
		y_train = np.load(y_train_save_path)
		y_test = np.load(y_test_save_path)
		state_len, num_classes = np.load(info_save_path)

		print("File loaded.")

		return state_len, num_classes, x_train, y_train, x_test, y_test

	print('Loading data...')
	(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
	print(len(x_train), 'train sequences')
	print(len(x_test), 'test sequences')

	print('Pad sequences (samples x time)')
	x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
	x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

	print('x_train shape:', x_train.shape)
	print('y_train shape:', y_train.shape)
	print('x_test shape:', x_test.shape)
	print('y_test shape:', y_test.shape)

	# train_max = np.max(x_train)
	# test_max =  np.max(x_test)
	# voc_num = (train_max, test_max)[test_max > train_max] + 1
	# state_len = maxlen * voc_num

	# #x_train = np.stack([keras.utils.to_categorical(x, voc_num).reshape(-1) for x in x_train], axis=0)
	# #x_test = np.stack([keras.utils.to_categorical(x, voc_num).reshape(-1) for x in x_test], axis=0)

	# x_train_ = np.zeros((x_train.shape[0], state_len))
	# x_test_ = np.zeros((x_test.shape[0], state_len))

	# print("Expand dimension of x_train...")
	# i = 0
	# for x in tqdm(x_train):
	# 	x = keras.utils.to_categorical(x, voc_num).reshape(-1)
	# 	x_train_[i, :] = x
	# 	i += 1

	# print("Expand dimension of x_test...")
	# i = 0
	# for x in tqdm(x_test):
	# 	x = keras.utils.to_categorical(x, voc_num).reshape(-1)
	# 	x_test_[i, :] = x
	# 	i += 1



	# x_train = x_train_.reshape(-1, 1, state_len)
	# x_test = x_test_.reshape(-1, 1, state_len)

	x_train = x_train.reshape(-1, 1, state_len)
	x_test = x_test.reshape(-1, 1, state_len)

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)	

	print('x_train shape:', x_train.shape)
	print('y_train shape:', y_train.shape)
	print('x_test shape:', x_test.shape)
	print('y_test shape:', y_test.shape)


	np.save(x_train_save_path, x_train)
	np.save(x_test_save_path, x_test)
	np.save(y_train_save_path, y_train)
	np.save(y_test_save_path, y_test)
	np.save(info_save_path, np.array([state_len, num_classes]))

	return state_len, num_classes, x_train, y_train, x_test, y_test


def load_cifar():
	num_classes = 10
	# input image dimensions
	img_rows, img_cols = 32, 32
	state_len = img_rows * img_cols

	(x_train_rgb, y_train), (x_test_rgb, y_test) = cifar10.load_data()

	x_train = np.zeros((x_train_rgb.shape[:-1]))
	x_test = np.zeros((x_test_rgb.shape[:-1]))

	for i, x in enumerate(x_train_rgb):
		x_train[i] = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

	for i, x in enumerate(x_test_rgb):
		x_test[i] = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

	if K.image_data_format() == 'channels_first':
	    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	    input_shape = (img_rows, img_cols, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	x_train = x_train.reshape(-1, 1, state_len)
	x_test = x_test.reshape(-1, 1, state_len)
	print(np.dot(x_train[0], x_train[0].T))
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return state_len, num_classes, x_train, y_train, x_test, y_test

def load_cifar_rgb():
	num_classes = 10
	# input image dimensions
	img_rows, img_cols, channels = 32, 32, 3
	state_len = img_rows * img_cols * channels

	(x_train, y_train), (x_test, y_test) = cifar10.load_data()


	if K.image_data_format() == 'channels_first':
	    x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
	    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
	    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
	    input_shape = (img_rows, img_cols, channels)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	x_train = x_train.reshape(-1, 1, state_len)
	x_test = x_test.reshape(-1, 1, state_len)
	print(np.dot(x_train[0], x_train[0].T))
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return state_len, num_classes, x_train, y_train, x_test, y_test
def load_breast_cancer():
	state_len = 9
	num_classes = 2
	# the data, split between train and test sets
	#(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x = np.load("dataset/x-bc.npy")
	y = np.load("dataset/y-bc.npy")

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=723)

	print(x_train.shape)
	print(x_test.shape)
	print(y_train.shape)
	print(y_test.shape)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= np.max(x_train)
	x_test /= np.max(x_train)

	#x_train = wrap_norm(x_train, np.linalg.norm(x_train, axis=1, keepdims=True))
	#x_test = wrap_norm(x_test, np.linalg.norm(x_test, axis=1, keepdims=True))
	#state_len += 1

	x_train = x_train.reshape(-1, 1, state_len)
	x_test = x_test.reshape(-1, 1, state_len)

	print(np.dot(x_train[0], x_train[0].T))

	print('x_train shape:', x_train.shape)
	print('y_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')


	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return state_len, num_classes, x_train, y_train, x_test, y_test

def load_wis():
	state_len = 30
	num_classes = 2
	# the data, split between train and test sets
	#(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x = np.load("dataset/x-wis.npy")
	y = np.load("dataset/y-wis.npy")

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=723)

	print(x_train.shape)
	print(x_test.shape)
	print(y_train.shape)
	print(y_test.shape)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= np.max(x_train)
	x_test /= np.max(x_train)

	#x_train = wrap_norm(x_train, np.linalg.norm(x_train, axis=1, keepdims=True))
	#x_test = wrap_norm(x_test, np.linalg.norm(x_test, axis=1, keepdims=True))
	#state_len += 1

	x_train = x_train.reshape(-1, 1, state_len)
	x_test = x_test.reshape(-1, 1, state_len)

	print(np.dot(x_train[0], x_train[0].T))

	print('x_train shape:', x_train.shape)
	print('y_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')


	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return state_len, num_classes, x_train, y_train, x_test, y_test


def load_wis_id():
	state_len = 31
	num_classes = 2
	# the data, split between train and test sets
	#(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x = np.load("dataset/x-wis-id.npy")
	y = np.load("dataset/y-wis-id.npy")

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=723)

	print(x_train.shape)
	print(x_test.shape)
	print(y_train.shape)
	print(y_test.shape)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= np.max(x_train)
	x_test /= np.max(x_train)

	#x_train = wrap_norm(x_train, np.linalg.norm(x_train, axis=1, keepdims=True))
	#x_test = wrap_norm(x_test, np.linalg.norm(x_test, axis=1, keepdims=True))
	#state_len += 1

	x_train = x_train.reshape(-1, 1, state_len)
	x_test = x_test.reshape(-1, 1, state_len)

	print(np.dot(x_train[0], x_train[0].T))

	print('x_train shape:', x_train.shape)
	print('y_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')


	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return state_len, num_classes, x_train, y_train, x_test, y_test

def load_cell():
	# state_len = 30
	# num_classes = 2
	# the data, split between train and test sets
	#(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# x = np.load("dataset/x-wis.npy")
	# y = np.load("dataset/y-wis.npy")

	data = loadmat("dataset/train_xy.mat")
	data = data["train_xy"]
	x = data[:,:-2]
	y = data[:,-1]
	y = y.astype(int)

	classes = np.unique(y)
	for i, c in enumerate(classes):
		y[y==c] = i

	bincnt = np.bincount(y)
	to_drop = np.where(bincnt < 256)[0]
	print(to_drop)
	print(y.shape)
	for d in to_drop:
		x = x[y!=d]
		y = y[y!=d]
	print(y.shape)

	classes = np.unique(y)
	for i, c in enumerate(classes):
		y[y==c] = i

	state_len = x.shape[-1]
	num_classes = len(np.unique(y))


	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=723)

	print(x_train.shape)
	print(x_test.shape)
	print(y_train.shape)
	print(y_test.shape)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= np.max(x_train)
	x_test /= np.max(x_train)

	#x_train = wrap_norm(x_train, np.linalg.norm(x_train, axis=1, keepdims=True))
	#x_test = wrap_norm(x_test, np.linalg.norm(x_test, axis=1, keepdims=True))
	#state_len += 1

	x_train = x_train.reshape(-1, 1, state_len)
	x_test = x_test.reshape(-1, 1, state_len)

	print(np.dot(x_train[0], x_train[0].T))

	print('x_train shape:', x_train.shape)
	print('y_train shape:', y_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')


	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return state_len, num_classes, x_train, y_train, x_test, y_test

def load_yinheng(sub_dir, is_augmented=False):

	aug = ("no_sampled", "sampled")[is_augmented]

	root = os.path.join("dataset/yinheng/", aug, sub_dir)
	x_train_path = os.path.join(root, "X_train.csv")
	x_test_path = os.path.join(root, "X_test.csv")
	y_train_path = os.path.join(root, "y_train.csv")
	y_test_path = os.path.join(root, "y_test.csv")

	with open(x_train_path, "r") as fd:
		x_ncols = len(fd.readline().split(","))
	with open(y_train_path, "r") as fd:
		y_ncols = len(fd.readline().split(","))

	x_train = np.loadtxt(x_train_path, delimiter=",", skiprows=1, usecols=range(1, x_ncols))
	x_test = np.loadtxt(x_test_path, delimiter=",", skiprows=1, usecols=range(1, x_ncols))
	y_train = np.loadtxt(y_train_path, delimiter=",", skiprows=1, usecols=range(1, y_ncols))
	y_test = np.loadtxt(y_test_path, delimiter=",", skiprows=1, usecols=range(1, y_ncols))

	x_train = x_train.reshape(x_train.shape[0], -1)
	x_test = x_test.reshape(x_test.shape[0], -1)

	if x_train.shape[-1] == 1:
		x_train = np.concatenate([np.zeros(x_train.shape), x_train], axis=1)
		x_test = np.concatenate([np.zeros(x_test.shape), x_test], axis=1)


	# x_train_col_sum = np.sum(x_train, axis=0)
	# x_train = x_train[:, np.where(x_train_col_sum != 0)]
	# x_test = x_test[:, np.where(x_train_col_sum != 0)]

	sample_ratio = 2

	x_train = np.repeat(x_train, (y_train == 1) * sample_ratio + 1, axis=0)
	y_train = np.repeat(y_train, (y_train == 1) * sample_ratio + 1, axis=0)

	# x_train = np.repeat(x_train, 10, axis=0)
	# y_train = np.repeat(y_train, 10, axis=0)


	p = np.random.permutation(x_train.shape[0])
	x_train = x_train[p]
	y_train = y_train[p]

	state_len = x_train.shape[-1]
	num_classes = len(np.unique(y_train))

	print(x_train.shape)
	print(x_test.shape)
	print(y_train.shape)
	print(y_test.shape)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	# x_train /= np.max(x_train)
	# x_test /= np.max(x_train)
	# x_train -= np.mean(x_train, axis=0, keepdims=True)
	# x_test -= np.mean(x_train, axis=0, keepdims=True)

	#x_train = wrap_norm(x_train, np.linalg.norm(x_train, axis=1, keepdims=True))
	#x_test = wrap_norm(x_test, np.linalg.norm(x_test, axis=1, keepdims=True))
	#state_len += 1

	x_train = x_train.reshape(-1, 1, state_len)
	x_test = x_test.reshape(-1, 1, state_len)

	print(np.dot(x_train[0], x_train[0].T))

	print('x_train shape:', x_train.shape)
	print('y_train shape:', y_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')


	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return state_len, num_classes, x_train, y_train, x_test, y_test

def load_titanic():
	pass

def load_red_mansions():
	pass

def load_shakespear():
	pass

def load_triple_gaussian():
	x1 = np.random.normal(-1, 0.5, 1000)
	x2 = np.random.normal(0, 0.5, 1000)
	x3 = np.random.normal(1, 0.5, 1000)

	x = np.concatenate([x1, x2, x3], axis=0)
	x = x.reshape(-1, 1)
	x = np.concatenate([np.zeros(x.shape), x*x, x], axis=1)

	y = np.concatenate([np.ones(x1.shape)*0, np.ones(x1.shape)*1, np.ones(x1.shape)*2,])

	# p = np.random.permutation(x.shape[0])

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=723)

	state_len = 3
	num_classes = 3

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= np.max(x_train)
	x_test /= np.max(x_train)
	x_train = x_train.reshape(-1, 1, state_len)
	x_test = x_test.reshape(-1, 1, state_len)

	print(np.dot(x_train[0], x_train[0].T))

	print('x_train shape:', x_train.shape)
	print('y_train shape:', y_train.shape)

	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return state_len, num_classes, x_train, y_train, x_test, y_test

def load_data_agent(data_tag):
	if data_tag == "mnist":
		return load_mnist()

	elif data_tag == "2d":
		return load_2d()

	elif data_tag == "3d":
		return load_3d()

	elif data_tag == "imdb":
		return load_imdb()

	elif data_tag == "imdb_noembed":
		return load_imdb_noembed()

	elif data_tag == "cifar":
		return load_cifar()

	elif data_tag == "cifar_rgb":
		return load_cifar_rgb()

	elif data_tag == "breast_cancer":
		return load_breast_cancer()

	elif data_tag == "wis":
		return load_wis()

	elif data_tag == "wis_id":
		return load_wis_id()

	elif data_tag == "titanic":
		return load_titanic()

	elif data_tag == "cell":
		return load_cell()
	elif data_tag == "triple_gaussian":
	 return load_triple_gaussian()

	elif data_tag.startswith("yinheng"):
		aug, sub_dir = data_tag.split("_")[1:]
		return load_yinheng(sub_dir, aug == "aug")

	else:
		raise ValueError("Unrecognized data tag. ")

def load_data(data_tag, centralized=False):
	state_len, num_classes, x_train, y_train, x_test, y_test = load_data_agent(data_tag)

	if centralized:
		x_train -= np.mean(x_train, axis=0, keepdims=True)
		x_test -= np.mean(x_test, axis=0, keepdims=True)

	return state_len, num_classes, x_train, y_train, x_test, y_test






