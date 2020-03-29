from __future__ import print_function
from datetime import datetime

import numpy as np

import tensorflow as tf
import keras
from keras import layers
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

MAGIC_CODE = "SIN_REAL_EIGEN"

# batch_size = 128
# num_classes = 10
# epochs = 12

# input image dimensions
# img_rows, img_cols = 28, 28
# state_len = 784

#GATE_FACTOR = K.constant(100)
# GATE_FACTOR = 50
GATE_FACTOR = 50
EIGEN_PERIOD = np.pi / 3
#EIGEN_PERIOD = 2 * np.pi
#EIGEN_PERIOD = 1
#EIGEN_PERIOD2 = 2

# HP_ORTHONORMAL = 1
# HP_EIGENDIST = 0.001
# HP_LESSSENS = 1


def get_time():
    now = datetime.now()

    now_stamp = now.strftime("%Y%m%d%H%M%S")
    return now_stamp


def wrap_norm(x, norm):
    return np.concatenate([x, norm], axis=1)

def raise_dim(data, regularizer=[], if_abs=False, neg_norm=False, complex_norm=False, frac_norm=False):
    pass

class EigenDist(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(EigenDist, self).__init__(**kwargs)

    def build(self, input_shape):
        #initializer = keras.initializers.RandomNormal()
        #initializer = keras.initializers.RandomUniform(minval=-3.14, maxval=3.14)
        initializer = keras.initializers.Zeros()
        #initializer = keras.initializers.Ones()
        self.kernel = self.add_weight(name="kernel",
                                      shape=(input_shape[2], self.output_dim),
                                      #initializer="uniform",
                                      initializer=initializer,
                                      trainable=True)
        super(EigenDist, self).build(input_shape)

    def call(self, inputs):
        inputs_shape = K.shape(self.kernel)
        # near orginal point loss
        #loss = K.mean(tf.math.reciprocal(K.epsilon() + K.pow(self.kernel, 2)))
        #self.add_loss(loss * 0.001)

        #loss = -K.mean(K.pow(self.kernel, 2))
        #self.add_loss(loss)

        #loss = K.mean(tf.keras.losses.MSE(K.pow(K.sin(EIGEN_PERIOD * self.kernel), 2), tf.ones(K.shape(self.kernel), dtype=self.kernel.dtype)))
        #self.add_loss(loss)
        loss = K.sum(tf.keras.losses.MSE(K.pow(K.sin(EIGEN_PERIOD * self.kernel), 2), tf.ones(K.shape(self.kernel), dtype=self.kernel.dtype)))
        self.add_loss(loss * HP_EIGENDIST)

        #noise = K.random_uniform(inputs_shape, minval=-0.05, maxval=0.05)
        #self.kernel += noise

        thresh = 1 + K.exp(-GATE_FACTOR * (K.sin(EIGEN_PERIOD * self.kernel)))
        # thresh = 1 + K.exp(-GATE_FACTOR * (tf.math.tan(EIGEN_PERIOD * self.kernel)))
        thresh = tf.math.reciprocal(thresh)
        #thresh *= 10

        #thresh_loss = K.sum(thresh)
        #thresh_loss = K.mean(K.pow(thresh, 2))
        #thresh_loss = K.sum(K.pow(K.sum(thresh, axis=1) - K.zeros((state_len,)), 1))
        #self.add_loss(K.pow(thresh_loss, 2))
        #self.add_loss(thresh_loss * 2)
        #self.add_loss(thresh_loss * HP_LESSSENS)

        thresh /= K.sum(thresh, axis=1, keepdims=True)
        #thresh /= K.pow(K.sum(thresh, axis=1, keepdims=True), 2)

        return tf.matmul(inputs, thresh)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        return {"output_dim": self.output_dim}


class Projection(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Projection, self).__init__(**kwargs)

    def build(self, input_shape):
        # initializer = keras.initializers.RandomNormal()
        # initializer = keras.initializers.RandomUniform()
        initializer = keras.initializers.Orthogonal(gain=1)
        #initializer = keras.initializers.Identity(gain=1)
        #initializer = keras.initializers.Zeros()
        self.kernel = self.add_weight(name="kernel",
                                      shape=(input_shape[2], self.output_dim),
                                      #initializer="uniform",
                                      initializer=initializer,
                                      trainable=True
                                      )
        super(Projection, self).build(input_shape)

    def call(self, inputs):
        inputs_shape = K.shape(inputs)
        # orthonomality
        mutual_projs = tf.matmul(K.transpose(self.kernel), self.kernel)
        mutual_probs = mutual_projs
        #mutual_projs = tf.matmul(tf.linalg.adjoint(self.kernel), self.kernel)
        #mutual_probs = tf.math.conj(mutual_projs) * mutual_projs
        eye = tf.eye(inputs_shape[2])
        #loss = K.mean(tf.keras.losses.MSE(eye, mutual_probs))
        #self.add_loss(loss * 10000)
        loss = K.sum(tf.keras.losses.MSE(eye, mutual_probs))
        self.add_loss(loss * HP_ORTHONORMAL)

        inputs = K.l2_normalize(inputs)
        outputs = K.pow(tf.matmul(inputs, self.kernel), 2)
        #var_loss = tf.math.reciprocal(0.1+K.mean(K.var(outputs, axis=2)))
        #self.add_loss(var_loss)

        return outputs
        #projs = tf.matmul(inputs, self.kernel)
        #return tf.math.conj(projs) * projs


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        return {"output_dim": self.output_dim}
        
if __name__ == "__main__":
    # # the data, split between train and test sets
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # if K.image_data_format() == 'channels_first':
    #     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    #     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    #     input_shape = (1, img_rows, img_cols)
    # else:
    #     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    #     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    #     input_shape = (img_rows, img_cols, 1)

    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    # x_train = x_train.reshape(-1, 1, state_len)
    # x_test = x_test.reshape(-1, 1, state_len)
    # print(np.dot(x_train[0], x_train[0].T))
    # print('x_train shape:', x_train.shape)
    # print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')

    # # convert class vectors to binary class matrices
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)

    # model = Sequential()
    # model.add(Projection(state_len, input_shape=(1, state_len)))
    # model.add(EigenDist(num_classes, input_shape=(1, state_len)))
    # model.add(Flatten())
    # model.summary()

    # model.compile(loss=keras.losses.categorical_crossentropy,
    # #model.compile(loss=keras.losses.mean_squared_error,
    #               optimizer=keras.optimizers.Adadelta(),
    #               metrics=['accuracy'])

    # history = model.fit(x_train, y_train,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           verbose=1,
    #           validation_data=(x_test, y_test))
    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

    # P = model.layers[0].get_weights()[0]
    # PP = P.T.dot(P)
    # print(PP)

    # L = model.layers[1].get_weights()[0]
    # LL = 1/(1+np.exp(L * -GATE_FACTOR))
    # print(LL)
    pass
