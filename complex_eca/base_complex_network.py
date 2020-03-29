from __future__ import print_function
import os
import shutil

import numpy as np

import tensorflow as tf
import keras
from keras import layers
# from keras.datasets import mnist
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Input, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import custom_object_scope
from keras import backend as K

import complex_eigen as ComplexEigen
# from complex_eigen import ProjectionRBF as Projection
from complex_eigen import Projection as Projection
from complex_eigen import Proj2Prob
from complex_eigen import RaiseDimension
from complex_eigen import RaiseDimensionNonQuadratic
from complex_eigen import RaiseDimensionIdentity
from complex_eigen import ReduceDimension
from complex_eigen import ReduceDimensionNonQuadratic
from complex_eigen import ReduceDimensionIdentity
from complex_eigen import FullConnectedNeuralNetwork, EigenDist, EigenDistApprox
from complex_eigen import GATE_FACTOR, MAGIC_CODE, wrap_norm, raise_dim, get_time
from load_data import load_data
from save_history import save_history
from other_models import lg, lda, qda, svm, compare_all, save_compare_result

batch_size = 128
epochs = 12

# state_len = 784

WORK_MAGIC_CODE = "PLAIN"

data_tag = "mnist"

# HP_ORTHONORMAL ~ 10 if initialized with orthogonaly
# HP_EIGENDIST ~ 10 / (state_len * num_classes)
#GATE_FACTOR = K.constant(100)
#GATE_FACTOR = 50
#ComplexEigen.GATE_FACTOR = GATE_FACTOR
ComplexEigen.HP_ORTHONORMAL = 0.1
ComplexEigen.HP_EIGENDIST = 0.001
ComplexEigen.HP_LESSSENS = 1

ECMM = EigenDistApprox

def categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=1e-10):
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)

    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)

        def _smooth_labels():
            num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - smoothing) + (smoothing / num_classes)

        y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)
    return K.categorical_crossentropy(y_true, y_pred+K.epsilon(), from_logits=from_logits)

def categorical_bernoulli_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=1e-10):
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)

    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)

        def _smooth_labels():
            num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - smoothing) + (smoothing / num_classes)

        y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)
    return K.categorical_crossentropy(y_true, y_pred+K.epsilon(), from_logits=from_logits)+K.categorical_crossentropy(1-y_true, 1-y_pred+K.epsilon(), from_logits=from_logits)


def build_model_do(
  state_len, num_classes, 
  activation="relu", 
  to_raise=True, to_reduce=True, 
  raise_quadratic=True, reduce_quadratic=True):
  global ECMM
  print("Building model...")


  if to_raise:
    raised_dim = 1024
    if not raise_quadratic:
      RaDO = RaiseDimensionNonQuadratic
    else:
      RaDO = RaiseDimension
  else:
    raised_dim = state_len
    RaDO = RaiseDimensionIdentity

  state_len1 = raised_dim

  if to_reduce:
    reduced_dim = 128
    if not reduce_quadratic:
      ReDO = ReduceDimensionNonQuadratic
    else:
      ReDO = ReduceDimension
  else:
    reduced_dim = state_len1
    ReDO = ReduceDimensionIdentity


  inputs = Input(shape=(2, state_len))
  # 1
  raise_dimension1 = RaDO(raised_dim, input_shape=(2, state_len))
  projection1 = Projection(state_len1, input_shape=(2, state_len1))
  proj2prob1 = Proj2Prob(state_len1, input_shape=(2, state_len1))
  eigen_dist1 = ECMM(num_classes, input_shape=(1, state_len1), name="1st_fold")
  flatten1 = Flatten(name="1st_fold")

  reduce_dimension1= ReDO(reduced_dim, input_shape=(2, state_len1))

  # 2
  state_len2 = reduced_dim
  projection2 = Projection(state_len2, input_shape=(2, state_len2))
  proj2prob2 = Proj2Prob(state_len2, input_shape=(2, state_len2))
  eigen_dist2 = ECMM(num_classes, input_shape=(1, state_len2), name="2nd_fold")
  flatten2 = Flatten(name="2nd_fold")

  # 1
  a0 = raise_dimension1(inputs)
  a1 = projection1(a0)
  a2 = proj2prob1(a1)
  # o1 = flatten1(eigen_dist1(a2))
  o1 = eigen_dist1(a2)
  a1 = reduce_dimension1(a1)

  # 2
  b1 = projection2(a1)
  b2 = proj2prob2(b1)
  # o2 = flatten2(eigen_dist2(b2))
  o2 = eigen_dist2(b2)

  model = Model(inputs, outputs=[o1, o2])



  return model

# def build_model_dnn(state_len, num_classes, activation="relu"):
#   global ECMM
#   print("Building model...")

#   reduced_dim = 128

#   inputs = Input(shape=(1, state_len))
#   # 1
#   state_len1 = state_len
#   projection1 = Projection(state_len1, input_shape=(1, state_len1))
#   proj2prob1 = Proj2Prob(state_len1, input_shape=(1, state_len1))
#   eigen_dist1 = ECMM(num_classes, input_shape=(1, state_len1))
#   flatten1 = Flatten(name="1st_fold")

#   # 2
#   flatten2_dense = Flatten()
#   state_len2 = reduced_dim
#   dense2 = Dense(state_len2, activation=activation)
#   reshape2 = Reshape((1, state_len2), input_shape=(state_len2,))
#   projection2 = Projection(state_len2, input_shape=(1, state_len2))
#   proj2prob2 = Proj2Prob(state_len2, input_shape=(1, state_len2))
#   eigen_dist2 = ECMM(num_classes, input_shape=(1, state_len2))
#   flatten2 = Flatten(name="2nd_fold")

#   # 1
#   a1 = projection1(inputs)
#   a2 = proj2prob1(a1)
#   o1 = flatten1(eigen_dist1(a2))

#   # 2
#   b1 = flatten2_dense(a1)
#   b2 = dense2(b1)
#   b3 = reshape2(b2)
#   b4 = projection2(b3)
#   b5 = proj2prob2(b4)
#   o2 = flatten2(eigen_dist2(b5))

#   model = Model(inputs, outputs=[o1, o2])

#   return model

def build_model(state_len, num_classes):
  global ECMM

  model = build_model_do(
    state_len, num_classes, 
    to_raise=True, to_reduce=False, 
    raise_quadratic=True, reduce_quadratic=True)

  # model = build_model_dnn(state_len, num_classes)

  model.summary()

  # ECMM = EigenDist
  ECMM = EigenDistApprox

  # model.compile(loss=keras.losses.categorical_crossentropy,
  # model.compile(loss=keras.losses.mean_squared_error,
  # model.compile(loss=categorical_bernoulli_crossentropy,
  # vanilla
  # model.compile(loss=[categorical_bernoulli_crossentropy, categorical_bernoulli_crossentropy],
  # approx
  model.compile(loss=[categorical_crossentropy, categorical_crossentropy],
                loss_weights=[0.5, 0.5],
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

  return model


def base_train(data_tag, to_train=False, is_bin=False):
  print("Unpacking data...")
  state_len, num_classes, x_train, y_train, x_test, y_test = load_data(data_tag)
  print(f"state_len: {state_len}, num_classes: {num_classes}")
  # x_train = np.expand_dims(x_train, axis=0)
  # x_test = np.expand_dims(x_test, axis=0)
  x_train = np.concatenate([x_train, np.zeros(x_train.shape)], axis=1)
  x_test = np.concatenate([x_test, np.zeros(x_test.shape)], axis=1)
  # print(x_train.dtype)
  # print(x_test.dtype)


  model = build_model(state_len, num_classes)
  print("Model built. ")

  time_stamp = get_time()
  print(time_stamp)
  model_save_root = f"checkpoints/{data_tag}/{MAGIC_CODE}"
  history_save_root = f"history/{data_tag}/{MAGIC_CODE}/{time_stamp}"
  os.makedirs(model_save_root, exist_ok=True)
  os.makedirs(history_save_root, exist_ok=True)

  model_basename = f"{MAGIC_CODE}-{data_tag}-{WORK_MAGIC_CODE}"
  model_save_path = f"{model_save_root}/{model_basename}-{time_stamp}.h5"
  model_universal = f"best_models/{model_basename}.h5"
  history = []
  if to_train:
    # earlystopper = EarlyStopping(patience=10, verbose=1, monitor="val_acc")
    # checkpointer = ModelCheckpoint(model_universal, verbose=1, save_best_only=True, monitor="val_acc")

    earlystopper = EarlyStopping(patience=10, verbose=1)
    checkpointer = ModelCheckpoint(model_universal, verbose=1, save_best_only=True)

    # reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min')
    history = model.fit(x_train, [y_train, y_train],
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, [y_test, y_test]), 
              # callbacks=[earlystopper, checkpointer, reduce_lr_loss])
              callbacks=[earlystopper, checkpointer])

  with custom_object_scope({
    "Projection":Projection, "Proj2Prob":Proj2Prob, 
    "EigenDist":EigenDist, "categorical_bernoulli_crossentropy":categorical_bernoulli_crossentropy, 
    "FullConnectedNeuralNetwork":FullConnectedNeuralNetwork, "categorical_crossentropy":categorical_crossentropy}):
    model.load_weights(model_universal)
    if to_train:
      model.save(model_save_path)

  score = model.evaluate(x_test, [y_test, y_test], verbose=0)
  print(score)
  # print('Test loss:', score[0])
  # print('Test loss 2:', score[1])
  # print('Test accuracy:', score[2])
  # print('Test accuracy 2:', score[3])

  # dataset = [x_train, y_train, x_test, y_test]
  # save_history(dataset, model, num_classes, history, data_tag, WORK_MAGIC_CODE, MAGIC_CODE, history_save_root, time_stamp)
  # cmp_res = compare_all(dataset, model, data_tag, WORK_MAGIC_CODE, MAGIC_CODE, time_stamp, is_bin=is_bin)
  # save_compare_result(cmp_res, data_tag, WORK_MAGIC_CODE, MAGIC_CODE, time_stamp)

  # print("Wait Nutstore to sync...")
  # import time
  # time.sleep(5)

  # shutil.copy(f"history_{data_tag}.txt", f"{history_save_root}/")

if __name__ == '__main__':
  base_train(data_tag, to_train=True, is_bin=False)




