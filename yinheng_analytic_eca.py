from __future__ import print_function
import os
import shutil

import numpy as np

from sklearn.metrics import accuracy_score

import tensorflow as tf
import keras
from keras import layers
# from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import custom_object_scope
from keras import backend as K

import real_eigen as RealEigen
# from real_eigen import ProjectionRBF as Projection
from real_eigen import Projection as Projection
from real_eigen import Proj2Prob, EigenDist, GATE_FACTOR, MAGIC_CODE, wrap_norm, raise_dim, get_time
from load_data import load_data
from save_history import save_history
from other_models import lg, lda, qda, svm, compare_all, save_compare_result

batch_size = 128
epochs = 12

# state_len = 784

WORK_MAGIC_CODE = "PLAIN"

data_tag = "yinheng_aug_1"

# HP_ORTHONORMAL ~ 10 if initialized with orthogonaly
# HP_EIGENDIST ~ 10 / (state_len * num_classes)
#GATE_FACTOR = K.constant(100)
#GATE_FACTOR = 50
#RealEigen.GATE_FACTOR = GATE_FACTOR
RealEigen.HP_ORTHONORMAL = 10
RealEigen.HP_EIGENDIST = 0.001
RealEigen.HP_LESSSENS = 1


def base_train(data_tag, to_train=False, is_bin=False):
  print("Unpacking data...")
  state_len, num_classes, x_train, y_train, x_test, y_test = load_data(data_tag)
  print(f"state_len: {state_len}, num_classes: {num_classes}")

  y_train = np.argmax(y_train,axis=1)
  y_train = y_train.reshape(-1, 1)
  y_test = np.argmax(y_test,axis=1)
  y_test = y_test.reshape(-1, 1)

  x_train = x_train.reshape(-1, state_len)
  x_norm = np.linalg.norm(x_train, axis=1, keepdims=True)
  x_train = x_train / x_norm

  x_test = x_test.reshape(-1, state_len)
  x_norm = np.linalg.norm(x_test, axis=1, keepdims=True)
  x_test = x_test / x_norm

  # h1 = (np.power(2, y_train.T) * x_train.T)
  h1 = y_train.T * x_train.T
  h1 = np.matmul(h1, x_train)
  h2 = np.matmul(x_train.T, x_train)
  h2 = np.linalg.inv(h2)

  H = np.matmul(h1, h2)

  print(H)

  w, v = np.linalg.eig(H)

  

  print(w)
  print(v)

  # pred = np.round(np.log2(np.sum(x_test.T * np.matmul(H, x_test.T),axis=0)))
  pred = np.round(np.sum(x_test.T * np.matmul(H, x_test.T),axis=0))
  acc = accuracy_score(y_test, pred)
  print(acc)




if __name__ == '__main__':
  base_train(data_tag, to_train=False, is_bin=False)


