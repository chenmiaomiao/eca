from __future__ import print_function
import os
import shutil

import numpy as np

import tensorflow as tf
import keras
from keras import layers
# from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import custom_object_scope
from keras import backend as K

import real_eigen as RealEigen
# from real_eigen import Projection, EigenDist, GATE_FACTOR, MAGIC_CODE, wrap_norm, raise_dim, get_time
# from load_data import load_data
# from save_history import save_history
# from other_models import lg, lda, qda, svm, compare_all, save_compare_result
import base_vanilla as base_model
from base_vanilla import base_train

batch_size = 128
epochs = 12

# state_len = 784

WORK_MAGIC_CODE = "PLAIN"

data_tag = "3d"

# HP_ORTHONORMAL ~ 10 if initialized with orthogonaly
# HP_EIGENDIST ~ 10 / (state_len * num_classes)
#GATE_FACTOR = K.constant(100)
#GATE_FACTOR = 50
#RealEigen.GATE_FACTOR = GATE_FACTOR
RealEigen.HP_ORTHONORMAL = 5
RealEigen.HP_EIGENDIST = 5
RealEigen.HP_LESSSENS = 1

base_model.WORK_MAGIC_CODE = WORK_MAGIC_CODE
base_model.batch_size = batch_size
base_model.epochs = epochs

# state_len, num_classes, x_train, y_train, x_test, y_test = load_data(data_tag)
# dataset = x_train, y_train, x_test, y_test

if __name__ == '__main__':
	base_train(data_tag, to_train=False, is_bin=True)


