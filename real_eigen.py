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

MAGIC_CODE = "FUSION_SIN_REAL_EIGEN"

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

def generate_class_weights(y_train):
    classes = np.unique(y_train)
    bincnt = np.bincount(y_train)
    weights = y_train.shape[0] / bincnt
    # weights /= np.sum(weights)

    weights = dict(zip(classes, weights))

    print(bincnt)
    print(weights)
    
    return weights


def get_time():
    now = datetime.now()

    now_stamp = now.strftime("%Y%m%d%H%M%S")
    return now_stamp


def wrap_norm(x, norm):
    return np.concatenate([x, norm], axis=1)

def raise_dim(data, regularizer=[], if_abs=False, neg_norm=False, complex_norm=False, frac_norm=False):
    pass

class RaiseDimensionNonQuadratic(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(RaiseDimensionNonQuadratic, self).__init__(**kwargs)

    def build(self, input_shape):
        initializer = keras.initializers.RandomNormal()
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[2]+1, self.output_dim-input_shape[2]),
            initializer=initializer,
            trainable=True,
            )

        super(RaiseDimensionNonQuadratic, self).build(input_shape)

    def call(self, inputs):
        inputs_shape = K.shape(inputs)

        auxilliary_dim = tf.ones((inputs_shape[0], inputs_shape[1], 1), dtype=inputs.dtype)
        inputs_aux = K.concatenate([auxilliary_dim, inputs], axis=2)

        extra_dim = tf.matmul(inputs_aux, self.kernel)

        outputs = K.relu(K.concatenate([inputs, extra_dim], axis=2))

        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        return {"output_dim":self.output_dim}

class RaiseDimensionIdentity(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(RaiseDimensionIdentity, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RaiseDimensionIdentity, self).build(input_shape)

    def call(self, inputs):

        return inputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        return {"output_dim":self.output_dim}

class RaiseDimensionSeries(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(RaiseDimensionSeries, self).__init__(**kwargs)

    def build(self, input_shape):
        initializer = keras.initializers.RandomNormal()
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[2]*2+1, self.output_dim-input_shape[2]),
            initializer=initializer,
            trainable=True,
            )

        super(RaiseDimensionSeries, self).build(input_shape)

    def call(self, inputs):
        inputs_shape = K.shape(inputs)

        auxilliary_dim = tf.ones((inputs_shape[0], inputs_shape[1], 1), dtype=inputs.dtype)
        inputs_squared = tf.pow(inputs, 2)
        inputs_aux = K.concatenate([auxilliary_dim, inputs, inputs_squared], axis=2)

        extra_dim = tf.matmul(inputs_aux, self.kernel)

        outputs = K.concatenate([inputs, extra_dim], axis=2)
        # outputs = K.relu(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        return {"output_dim":self.output_dim}

class RaiseDimension(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(RaiseDimension, self).__init__(**kwargs)

    def build(self, input_shape):
        initializer = keras.initializers.RandomNormal()
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[2]+1, self.output_dim-input_shape[2]),
            initializer=initializer,
            trainable=True,
            )

        super(RaiseDimension, self).build(input_shape)

    def call(self, inputs):
        inputs_shape = K.shape(inputs)

        auxilliary_dim = tf.ones((inputs_shape[0], inputs_shape[1], 1), dtype=inputs.dtype)
        inputs_squared = tf.pow(inputs, 2)
        inputs_aux = K.concatenate([auxilliary_dim, inputs_squared], axis=2)

        kernel = tf.pow(self.kernel, 2)

        extra_dim = K.sqrt(tf.matmul(inputs_aux, kernel))

        outputs = K.concatenate([inputs, extra_dim], axis=2)
        # outputs = K.relu(outputs)
        # outputs = extra_dim

        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        return {"output_dim":self.output_dim}

class ReduceDimensionNonQuadratic(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ReduceDimensionNonQuadratic, self).__init__(**kwargs)

    def build(self, input_shape):
        initializer = keras.initializers.RandomNormal()
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[2]+1, self.output_dim),
            initializer=initializer,
            trainable=True
            )

        super(ReduceDimensionNonQuadratic, self).build(input_shape)

    def call(self, inputs):
        inputs_shape = K.shape(inputs)

        auxilliary_dim = tf.ones((inputs_shape[0], inputs_shape[1], 1), dtype=inputs.dtype)
        inputs_aux = K.concatenate([auxilliary_dim, inputs], axis=2)

        outputs = K.relu(tf.matmul(inputs_aux, self.kernel))

        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        return {"output_dim":self.output_dim}

class ReduceDimensionIdentity(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ReduceDimensionIdentity, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReduceDimensionIdentity, self).build(input_shape)

    def call(self, inputs):

        return inputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        return {"output_dim":self.output_dim}

class ReduceDimension(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ReduceDimension, self).__init__(**kwargs)

    def build(self, input_shape):
        initializer = keras.initializers.RandomNormal()
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[2]+1, self.output_dim),
            initializer=initializer,
            trainable=True
            )

        super(ReduceDimension, self).build(input_shape)

    def call(self, inputs):
        inputs_shape = K.shape(inputs)

        auxilliary_dim = tf.ones((inputs_shape[0], inputs_shape[1], 1), dtype=inputs.dtype)
        inputs_aux = K.concatenate([auxilliary_dim, inputs], axis=2)

        kernel = tf.pow(self.kernel, 2)

        outputs = K.sqrt(tf.matmul(inputs_aux, kernel))

        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        return {"output_dim":self.output_dim}



class Softmax(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Softmax, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Softmax, self).build(input_shape)

    def call(self, inputs):
        outputs = K.softmax(inputs, axis=-1)

        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        return {"output_dim": self.output_dim}

class FullConnectedNeuralNetwork(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(FullConnectedNeuralNetwork, self).__init__(**kwargs)

    def build(self, input_shape):
        initializer = keras.initializers.RandomNormal()
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[2]+1, self.output_dim),
            initializer=initializer,
            trainable=True
        )
        super(FullConnectedNeuralNetwork, self).build(input_shape)

    def call(self, inputs):
        inputs_shape = K.shape(inputs)
        auxilliary_dim = tf.ones((inputs_shape[0], inputs_shape[1], 1), dtype=inputs.dtype)
        auxilliary_inputs = K.concatenate([auxilliary_dim, inputs], axis=2)

        outputs = tf.matmul(auxilliary_inputs, self.kernel)

        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        return {"output_dim": self.output_dim}

class EigenDistApprox(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(EigenDistApprox, self).__init__(**kwargs)

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
        super(EigenDistApprox, self).build(input_shape)

    def call(self, inputs):
        # kernel_shape = K.shape(self.kernel)
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
        # kernel_shape = K.shape(self.kernel)
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

        # thresh /= K.sum(thresh, axis=1, keepdims=True)
        #thresh /= K.pow(K.sum(thresh, axis=1, keepdims=True), 2)

        return tf.matmul(inputs, thresh)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        return {"output_dim": self.output_dim}

class Proj2Prob(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Proj2Prob, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Proj2Prob, self).build(input_shape)

    def call(self, inputs):
        outputs = K.pow(inputs, 2)

        return outputs

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
        # outputs = K.pow(tf.matmul(inputs, self.kernel), 2)
        outputs = tf.matmul(inputs, self.kernel)
        #var_loss = tf.math.reciprocal(0.1+K.mean(K.var(outputs, axis=2)))
        #self.add_loss(var_loss)

        return outputs
        #projs = tf.matmul(inputs, self.kernel)
        #return tf.math.conj(projs) * projs


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        return {"output_dim": self.output_dim}

class ProjectionRBF(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ProjectionRBF, self).__init__(**kwargs)

    def build(self, input_shape):
        initializer = keras.initializers.Orthogonal(gain=1)
        # initializer_rbf = keras.initializers.RandomNormal()

        self.kernel = self.add_weight(name="kernel",
                                      shape=(input_shape[2], self.output_dim),
                                      #initializer="uniform",
                                      initializer=initializer,
                                      trainable=True
                                      )

        # self.rbf_weight = self.add_weight(name="rbf_weight",
        #                               shape=(input_shape[2], self.output_dim),
        #                               #initializer="uniform",
        #                               initializer=initializer_rbf,
        #                               trainable=True
        #                               )
        super(ProjectionRBF, self).build(input_shape)

    def call(self, inputs):
        inputs_shape = K.shape(inputs)

        # orthonomality

        # # linear
        # mutual_projs = tf.matmul(K.transpose(self.kernel), self.kernel)

        # rbf
        expand_kernel = K.expand_dims(self.kernel, axis=2)
        diff = expand_kernel - self.kernel
        diff_norm = tf.norm(diff, axis=1)
        diff_norm = K.pow(diff_norm, 2)
        mutual_projs = K.exp(-diff_norm)

        # # poly
        # inner_prod = tf.matmul(K.transpose(self.kernel), self.kernel)
        # inner_prod += 1
        # mutual_projs = K.pow(inner_prod, 3)
        # mutual_probs = mutual_projs


        mutual_probs = mutual_projs

        eye = tf.eye(inputs_shape[2])
        loss = K.sum(tf.keras.losses.MSE(eye, mutual_probs))
        self.add_loss(loss * HP_ORTHONORMAL)

        inputs = K.l2_normalize(inputs)

        # # linear
        # outputs = tf.matmul(inputs, self.kernel)

        # rbf 
        diff = inputs - K.transpose(self.kernel)
        diff_norm = tf.norm(diff, axis=2)
        diff_norm = K.pow(diff_norm, 2)
        diff_norm = K.reshape(diff_norm, inputs_shape)
        # diff_norm = K.expand_dims(diff_norm, axis=1)
        outputs = K.exp(-diff_norm)

        # # poly
        # inner_prod = tf.matmul(inputs, self.kernel)
        # inner_prod += 1
        # outputs = K.pow(inner_prod, 3)

        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        return {"output_dim": self.output_dim}
        
if __name__ == "__main__":
    pass
