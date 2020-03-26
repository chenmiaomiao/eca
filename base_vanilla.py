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

data_tag = "mnist"

# HP_ORTHONORMAL ~ 10 if initialized with orthogonaly
# HP_EIGENDIST ~ 10 / (state_len * num_classes)
#GATE_FACTOR = K.constant(100)
#GATE_FACTOR = 50
#RealEigen.GATE_FACTOR = GATE_FACTOR
RealEigen.HP_ORTHONORMAL = 10
RealEigen.HP_EIGENDIST = 0.001
RealEigen.HP_LESSSENS = 1

def categorical_bernoulli_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)

    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)

        def _smooth_labels():
            num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - smoothing) + (smoothing / num_classes)

        y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)
    return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)+K.categorical_crossentropy(1-y_true, 1-y_pred, from_logits=from_logits)


def build_model(state_len, num_classes):
	print("Building model...")

	model = Sequential()
	model.add(Projection(state_len, input_shape=(1, state_len)))
	model.add(Proj2Prob(state_len, input_shape=(1, state_len)))
	model.add(EigenDist(num_classes, input_shape=(1, state_len)))
	model.add(Flatten())
	model.summary()

	# model.compile(loss=keras.losses.categorical_crossentropy,
	# model.compile(loss=keras.losses.mean_squared_error,
	model.compile(loss=categorical_bernoulli_crossentropy,
	              optimizer=keras.optimizers.Adadelta(),
	              metrics=['accuracy'])

	return model

def base_train(data_tag, to_train=False, is_bin=False):
	print("Unpacking data...")
	state_len, num_classes, x_train, y_train, x_test, y_test = load_data(data_tag)
	print(f"state_len: {state_len}, num_classes: {num_classes}")

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
		history = model.fit(x_train, y_train,
		          batch_size=batch_size,
		          epochs=epochs,
		          verbose=1,
		          validation_data=(x_test, y_test), 
		          # callbacks=[earlystopper, checkpointer, reduce_lr_loss])
		          callbacks=[earlystopper, checkpointer])

	with custom_object_scope({"Projection":Projection, "Proj2Prob":Proj2Prob, "EigenDist": EigenDist, "categorical_bernoulli_crossentropy":categorical_bernoulli_crossentropy}):
		model.load_weights(model_universal)
		if to_train:
			model.save(model_save_path)

	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	dataset = [x_train, y_train, x_test, y_test]
	save_history(dataset, model, num_classes, history, data_tag, WORK_MAGIC_CODE, MAGIC_CODE, history_save_root, time_stamp)
	cmp_res = compare_all(dataset, model, data_tag, WORK_MAGIC_CODE, MAGIC_CODE, time_stamp, is_bin=is_bin)
	save_compare_result(cmp_res, data_tag, WORK_MAGIC_CODE, MAGIC_CODE, time_stamp)

	print("Wait Nutstore to sync...")
	import time
	time.sleep(5)

	shutil.copy(f"history_{data_tag}.txt", f"{history_save_root}/")

if __name__ == '__main__':
	base_train(data_tag, to_train=False, is_bin=False)


