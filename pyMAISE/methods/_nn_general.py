import collections
from math import ceil

import kerastuner as kt
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from kerastuner import HyperModel
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

import pyMAISE.settings as settings


class DenseLayerHyperModel(HyperModel):
    def __init__(self, parameters: dict = None):
        self._input_dim = None

        # If there is dense layers
        self._dense = None
        self._num_nodes = 0
        self._activation = None
        self._use_bias = None
        self._kernel_initializer = "glorot_uniform"
        self._bias_initializer = "zeros"
        self._kernel_regularizer = None
        self._bias_regularizer = None
        self._activity_regularizer = None
        self._kernel_constraint = None
        self._bias_constraint = None

        if parameters != None:
            for key, value in parameters.items():
                setattr(self, key, value)

    def build(self, hp):
        if isinstance(self._num_nodes, list):
            num_nodes = hp.Int(
                "Units",
                min_value=self._num_nodes[0],
                max_value=self._num_nodes[1],
                step=self._num_nodes[2],
            )
        else:
            num_nodes = self._num_nodes
        if self._input_dim is None:
            return Dense(
                units=num_nodes,
                activation=self._activation,
                use_bias=self._use_bias,
                kernel_initializer=self._kernel_initializer,
                bias_initializer=self._bias_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                activity_regularizer=self._activity_regularizer,
                kernel_constraint=self._kernel_constraint,
                bias_constraint=self._bias_constraint,
            )
        else:
            return Dense(
                units=num_nodes,
                activation=self._activation,
                use_bias=self._use_bias,
                kernel_initializer=self._kernel_initializer,
                bias_initializer=self._bias_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                activity_regularizer=self._activity_regularizer,
                kernel_constraint=self._kernel_constraint,
                bias_constraint=self._bias_constraint,
                input_dim=self._input_dim,
            )

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def activation(self):
        return self._activation

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self._num_nodes = num_nodes

    @activation.setter
    def activation(self, activation):
        self._activation = activation

    @input_dim.setter
    def input_dim(self, input_dim):
        self._input_dim = input_dim


class nnHyperModel(HyperModel):
    def __init__(
        self,
        structural_hyperparameters: dict = None,
        optimizer_hyperparameters: dict = None,
    ):
        # Contains model layers along with each layers information
        # The hyperparameters that define the overall architecture-----------------------------------------
        # ie hidden units, number of layers, etc
        self._structural_hyperparameters = structural_hyperparameters

        # The hyperparameters that influence the speed and quality of training -------------------------
        # ie learning rate, type of optimizer, batch size, number of epochs
        self._learning_rate = 0.001
        self._optimizer = None
        self._batch_size = 10
        self._epochs = 10
        # Optimizer Info if adam
        self._optimizer = None
        self._learning_rate = 0.001
        self._beta_1 = 0.9
        self._beta_2 = 0.999
        self._epsilon = 1e-07
        self._amsgrad = False
        self._weight_decay = None
        self._clipnorm = None
        self._clipvalue = None
        self._global_clipnorm = None
        self._use_ema = False
        self._ema_momentum = 0.99
        self._ema_overwrite_frequency = None
        self._jit_compile = True

        # Model Compile
        self._loss = None
        self._metrics = None
        self._loss_weights = None
        self._weighted_metrics = None
        self._run_eagerly = None
        self._steps_per_execution = None
        self._pss_evaluation_shards = 0

        if optimizer_hyperparameters != None:
            for key, value in optimizer_hyperparameters.items():
                setattr(self, key, value)

        print("--Checking Initialization--")
        print("opt = ", self._optimizer)
        print("loss = ", self._loss)

    def build(self, hp):
        # Define Keras Model
        model = Sequential()

        # Iterating though archetecture and building a model from each layer
        for key in self._structural_hyperparameters[
            "structural_hyperparameters"
        ].keys():
            # print("archeteture = ", self._structural_hyperparameters["structural_hyperparameters"].keys())
            # print("key =", key)
            # Initializing a unique layer with information
            layer = DenseLayerHyperModel(
                self._structural_hyperparameters["structural_hyperparameters"][str(key)]
            )
            # adding the layer to the model
            if key == "dense" or "dense_input" or "dense_output":
                model.add(layer.build(hp))

        # Optimizer if not given one
        if self._optimizer == "adam":
            self._optimizer = Adam(
                learning_rate=self._learning_rate,
                beta_1=self._beta_1,
                beta_2=self._beta_2,
                epsilon=self._epsilon,
                amsgrad=self._amsgrad,
                weight_decay=self._weight_decay,
                clipnorm=self._clipnorm,
                clipvalue=self._clipvalue,
                global_clipnorm=self._global_clipnorm,
                use_ema=self._use_ema,
                ema_momentum=self._ema_momentum,
                ema_overwrite_frequency=self._ema_overwrite_frequency,
                jit_compile=self._jit_compile,
            )

        # Compile Model
        model.compile(
            optimizer=self._optimizer,
            loss=self._loss,
            metrics=self._metrics,
            loss_weights=self._loss_weights,
            weighted_metrics=self._weighted_metrics,
            run_eagerly=self._run_eagerly,
        )
        return model

    # =============================================================
    # Getters
    @property
    def optimizer(self):
        return self._optimizer

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def epochs(self):
        return self._epochs

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def loss(self):
        return self._loss

    @property
    def metrics(self):
        return self._metrics

    @property
    def warm_start(self):
        return self._warm_start

    @property
    def jit_compile(self):
        return self._jit_compile

    # ======================================================
    # Setters
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    @epochs.setter
    def epochs(self, epochs):
        self._epochs = epochs

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    @loss.setter
    def loss(self, loss):
        self._loss = loss

    @metrics.setter
    def metrics(self, metrics):
        self._metrics = metrics

    @warm_start.setter
    def warm_start(self, warm_start):
        self._warm_start = warm_start

    @jit_compile.setter
    def jit_compile(self, jit_compile):
        self._jit_compile = jit_compile
