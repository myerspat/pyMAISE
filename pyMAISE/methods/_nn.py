from pyMAISE.methods._nn_wrapper import NeuralNetsWrapper


class NeuralNetsRegression:
    def __init__(self, parameters: dict = None):
        # Model parameters
        # Sequential
        self._num_layers = None
        self._name = None
        self._dropout = False
        self._rate = 0.5
        self._loss = None
        self._metrics = None
        self._loss_weights = None
        self._weighted_metrics = None
        self._run_eagerly = False
        self._steps_per_execution = None
        self._jit_compile = True  # Used for both compile and adam
        self._batch_size = None
        self._validation_batch_size = None
        self._shuffle = True
        self._callbacks = None
        self._validation_split = 0.0
        self._epochs = 1
        self._warm_start = False

        # Starting Layer
        self._start_num_nodes = None
        self._start_activation = None
        self._start_use_bias = True
        self._start_kernel_initializer = "glorot_uniform"
        self._start_bias_initializer = "zeros"
        self._start_kernel_regularizer = None
        self._start_bias_regularizer = None
        self._start_activity_regularizer = None
        self._start_kernel_constraint = None
        self._start_bias_constraint = None
        self._input_dim = None

        # Middle Layers
        self._mid_num_node_strategy = None
        self._mid_activation = None
        self._mid_use_bias = True
        self._mid_kernel_initializer = "glorot_uniform"
        self._mid_bias_initializer = "zeros"
        self._mid_kernel_regularizer = None
        self._mid_bias_regularizer = None
        self._mid_activity_regularizer = None
        self._mid_kernel_constraint = None
        self._mid_bias_constraint = None

        # Ending Layer
        self._end_num_nodes = None
        self._end_activation = None
        self._end_use_bias = True
        self._end_kernel_initializer = "glorot_uniform"
        self._end_bias_initializer = "zeros"
        self._end_kernel_regularizer = None
        self._end_bias_regularizer = None
        self._end_activity_regularizer = None
        self._end_kernel_constraint = None
        self._end_bias_constraint = None

        # Optimizer
        self._optimizer = "adam"

        # Adam
        self._learning_rate = 0.001
        self._beta_1 = 0.9
        self._beta_2 = 0.999
        self._epsilon = 1e-7
        self._amsgrad = False
        self._clipnorm = None
        self._clipvalue = None
        self._global_clipnorm = None

        # Change if user provided changes in dictionary
        if parameters != None:
            for key, value in parameters.items():
                setattr(self, key, value)

    # ===========================================================
    # Methods
    def regressor(self):
        return NeuralNetsWrapper(
            num_layers=self._num_layers,
            name=self._name,
            dropout=self._dropout,
            rate=self._rate,
            loss=self._loss,
            metrics=self._metrics,
            loss_weights=self._loss_weights,
            weighted_metrics=self._weighted_metrics,
            run_eagerly=self._run_eagerly,
            steps_per_execution=self._steps_per_execution,
            batch_size=self._batch_size,
            validation_batch_size=self._validation_batch_size,
            shuffle=self._shuffle,
            callbacks=self._callbacks,
            validation_split=self._validation_split,
            epochs=self._epochs,
            warm_start=self._warm_start,
            start_num_nodes=self._start_num_nodes,
            start_activation=self._start_activation,
            start_use_bias=self._start_use_bias,
            start_kernel_initializer=self._start_kernel_initializer,
            start_bias_initializer=self._start_bias_initializer,
            start_kernel_regularizer=self._start_kernel_regularizer,
            start_bias_regularizer=self._start_bias_regularizer,
            start_activity_regularizer=self._start_activity_regularizer,
            start_kernel_constraint=self._start_kernel_constraint,
            start_bias_constraint=self._start_bias_constraint,
            input_dim=self._input_dim,
            mid_num_node_strategy=self._mid_num_node_strategy,
            mid_activation=self._mid_activation,
            mid_use_bias=self._mid_use_bias,
            mid_kernel_initializer=self._mid_kernel_initializer,
            mid_bias_initializer=self._mid_bias_initializer,
            mid_kernel_regularizer=self._mid_kernel_regularizer,
            mid_bias_regularizer=self._mid_bias_regularizer,
            mid_activity_regularizer=self._mid_activity_regularizer,
            mid_kernel_constraint=self._mid_kernel_constraint,
            mid_bias_constraint=self._mid_bias_constraint,
            end_num_nodes=self._end_num_nodes,
            end_activation=self._end_activation,
            end_use_bias=self._end_use_bias,
            end_kernel_initializer=self._end_kernel_initializer,
            end_bias_initializer=self._end_bias_initializer,
            end_kernel_regularizer=self._end_kernel_regularizer,
            end_bias_regularizer=self._end_bias_regularizer,
            end_activity_regularizer=self._end_activity_regularizer,
            end_kernel_constraint=self._end_kernel_constraint,
            end_bias_constraint=self._end_bias_constraint,
            optimizer=self._optimizer,
            learning_rate=self._learning_rate,
            beta_1=self._beta_1,
            beta_2=self._beta_2,
            epsilon=self._epsilon,
            amsgrad=self._amsgrad,
            clipnorm=self._clipnorm,
            clipvalue=self._clipvalue,
            global_clipnorm=self._global_clipnorm,
            jit_compile=self._jit_compile,
        )

    # ===========================================================
    # Getters
    @property
    def num_layers(self):
        return self._num_layers

    @property
    def name(self):
        return self._name

    @property
    def dropout(self):
        return self._dropout

    @property
    def rate(self):
        return self._rate

    @property
    def loss(self):
        return self._loss

    @property
    def metrics(self):
        return self._metrics

    @property
    def loss_weights(self):
        return self._loss_weights

    @property
    def run_eagerly(self):
        return self._run_eagerly

    @property
    def steps_per_execution(self):
        return self._steps_per_execution

    @property
    def jit_compile(self):
        return self._jit_compile

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def validation_batch_size(self):
        return self._validation_batch_size

    @property
    def shuffle(self):
        return self._shuffle

    @property
    def callbacks(self):
        return self._callbacks

    @property
    def validation_split(self):
        return self._validation_split

    @property
    def epochs(self):
        return self._epochs

    @property
    def warm_start(self):
        return self._warm_start

    @property
    def start_num_nodes(self):
        return self._start_num_nodes

    @property
    def start_activation(self):
        return self._start_activation

    @property
    def start_use_bias(self):
        return self._start_use_bias

    @property
    def start_kernel_initializer(self):
        return self._start_kernel_initializer

    @property
    def start_bias_initializer(self):
        return self._start_bias_initializer

    @property
    def start_kernel_regularizer(self):
        return self._start_kernel_regularizer

    @property
    def start_bias_regularizer(self):
        return self._start_bias_regularizer

    @property
    def start_activity_regularizer(self):
        return self._start_activity_regularizer

    @property
    def start_kernel_constraint(self):
        return self._start_kernel_constraint

    @property
    def start_bias_constraint(self):
        return self._start_bias_constraint

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def mid_num_node_strategy(self):
        return self._mid_num_node_strategy

    @property
    def mid_activation(self):
        return self._mid_activation

    @property
    def mid_use_bias(self):
        return self._mid_use_bias

    @property
    def mid_kernel_initializer(self):
        return self._mid_kernel_initializer

    @property
    def mid_bias_initializer(self):
        return self._mid_bias_initializer

    @property
    def mid_kernel_regularizer(self):
        return self._mid_kernel_regularizer

    @property
    def mid_bias_regularizer(self):
        return self._mid_bias_regularizer

    @property
    def mid_activity_regularizer(self):
        return self._mid_activity_regularizer

    @property
    def mid_kernel_constraint(self):
        return self._mid_kernel_constraint

    @property
    def mid_bias_constraint(self):
        return self._mid_bias_constraint

    @property
    def end_num_nodes(self):
        return self._end_num_nodes

    @property
    def end_activation(self):
        return self._end_activation

    @property
    def end_use_bias(self):
        return self._end_use_bias

    @property
    def end_kernel_initializer(self):
        return self._end_kernel_initializer

    @property
    def end_bias_initializer(self):
        return self._end_bias_initializer

    @property
    def end_kernel_regularizer(self):
        return self._end_kernel_regularizer

    @property
    def end_bias_regularizer(self):
        return self._end_bias_regularizer

    @property
    def end_activity_regularizer(self):
        return self._end_activity_regularizer

    @property
    def end_kernel_constraint(self):
        return self._start_end_constraint

    @property
    def end_bias_constraint(self):
        return self._end_bias_constraint

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def beta_1(self):
        return self._beta_1

    @property
    def beta_2(self):
        return self._beta_2

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def amsgrad(self):
        return self._amsgrad

    @property
    def clipnorm(self):
        return self._clipnorm

    @property
    def clipvalue(self):
        return self._clipvalue

    @property
    def global_clipnorm(self):
        return self._global_clipnorm

    # ===========================================================
    # Setters
    @num_layers.setter
    def num_layers(self, num_layers):
        self._num_layers = num_layers

    @name.setter
    def name(self, name):
        self._name = name

    @dropout.setter
    def dropout(self, dropout):
        self._dropout = dropout

    @rate.setter
    def rate(self, rate):
        self._rate = rate

    @loss.setter
    def loss(self, loss):
        self._loss = loss

    @metrics.setter
    def metrics(self, metrics):
        self._metrics = metrics

    @loss_weights.setter
    def loss_weights(self, loss_weights):
        self._loss_weights = loss_weights

    @run_eagerly.setter
    def run_eagerly(self, run_eagerly):
        self._run_eagerly = run_eagerly

    @steps_per_execution.setter
    def steps_per_execution(self, steps_per_execution):
        self._steps_per_execution = steps_per_execution

    @jit_compile.setter
    def jit_compile(self, jit_compile):
        self._jit_compile = jit_compile

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    @validation_batch_size.setter
    def validation_batch_size(self, validation_batch_size):
        self._validation_batch_size = validation_batch_size

    @shuffle.setter
    def shuffle(self, shuffle):
        self._shuffle = shuffle

    @callbacks.setter
    def callbacks(self, callbacks):
        self._callbacks = callbacks

    @validation_split.setter
    def validation_split(self, validation_split):
        self._validation_split = validation_split

    @epochs.setter
    def epochs(self, epochs):
        self._epochs = epochs

    @warm_start.setter
    def warm_start(self, warm_start):
        self._warm_start = warm_start

    @start_num_nodes.setter
    def start_num_nodes(self, start_num_nodes):
        self._start_num_nodes = start_num_nodes

    @start_activation.setter
    def start_activation(self, start_activation):
        self._start_activation = start_activation

    @start_use_bias.setter
    def start_use_bias(self, start_use_bias):
        self._start_use_bias = start_use_bias

    @start_kernel_initializer.setter
    def start_kernel_initializer(self, start_kernel_initializer):
        self._start_kernel_initializer = start_kernel_initializer

    @start_bias_initializer.setter
    def start_bias_initializer(self, start_bias_initializer):
        self._start_bias_initializer = start_bias_initializer

    @start_kernel_regularizer.setter
    def start_kernel_regularizer(self, start_kernel_regularizer):
        self._start_kernel_regularizer = start_kernel_regularizer

    @start_bias_regularizer.setter
    def start_bias_regularizer(self, start_bias_regularizer):
        self._start_bias_regularizer = start_bias_regularizer

    @start_activity_regularizer.setter
    def start_activity_regularizer(self, start_activity_regularizer):
        self._start_activity_regularizer = start_activity_regularizer

    @start_kernel_constraint.setter
    def start_kernel_constraint(self, start_kernel_constraint):
        self._start_kernel_constraint = start_kernel_constraint

    @start_bias_constraint.setter
    def start_bias_constraint(self, start_bias_constraint):
        self._start_bias_constraint = start_bias_constraint

    @input_dim.setter
    def input_dim(self, input_dim):
        self._input_dim = input_dim

    @mid_num_node_strategy.setter
    def mid_num_node_strategy(self, mid_num_node_strategy):
        assert isinstance(mid_num_node_strategy, str) or isinstance(
            mid_num_node_strategy, callable
        )
        self._mid_num_node_strategy = mid_num_node_strategy

    @mid_activation.setter
    def mid_activation(self, mid_activation):
        self._mid_activation = mid_activation

    @mid_use_bias.setter
    def mid_use_bias(self, mid_use_bias):
        self._mid_use_bias = mid_use_bias

    @mid_kernel_initializer.setter
    def mid_kernel_initializer(self, mid_kernel_initializer):
        self._mid_kernel_initializer = mid_kernel_initializer

    @mid_bias_initializer.setter
    def mid_bias_initializer(self, mid_bias_initializer):
        self._mid_bias_initializer = mid_bias_initializer

    @mid_kernel_regularizer.setter
    def mid_kernel_regularizer(self, mid_kernel_regularizer):
        self._mid_kernel_regularizer = mid_kernel_regularizer

    @mid_bias_regularizer.setter
    def mid_bias_regularizer(self, mid_bias_regularizer):
        self._mid_bias_regularizer = mid_bias_regularizer

    @mid_activity_regularizer.setter
    def mid_activity_regularizer(self, mid_activity_regularizer):
        self._mid_activity_regularizer = mid_activity_regularizer

    @mid_kernel_constraint.setter
    def mid_kernel_constraint(self, mid_kernel_constraint):
        self._mid_kernel_constraint = mid_kernel_constraint

    @mid_bias_constraint.setter
    def mid_bias_constraint(self, mid_bias_constraint):
        self._mid_bias_constraint = mid_bias_constraint

    @end_num_nodes.setter
    def end_num_nodes(self, end_num_nodes):
        self._end_num_nodes = end_num_nodes

    @end_activation.setter
    def end_activation(self, end_activation):
        self._end_activation = end_activation

    @end_use_bias.setter
    def end_use_bias(self, end_use_bias):
        self._end_use_bias = end_use_bias

    @end_kernel_initializer.setter
    def end_kernel_initializer(self, end_kernel_initializer):
        self._end_kernel_initializer = end_kernel_initializer

    @end_bias_initializer.setter
    def end_bias_initializer(self, end_bias_initializer):
        self._end_bias_initializer = end_bias_initializer

    @end_kernel_regularizer.setter
    def end_kernel_regularizer(self, end_kernel_regularizer):
        self._end_kernel_regularizer = end_kernel_regularizer

    @end_bias_regularizer.setter
    def end_bias_regularizer(self, end_bias_regularizer):
        self._end_bias_regularizer = end_bias_regularizer

    @end_activity_regularizer.setter
    def end_activity_regularizer(self, end_activity_regularizer):
        self._end_activity_regularizer = end_activity_regularizer

    @end_kernel_constraint.setter
    def end_kernel_constraint(self, end_kernel_constraint):
        self._start_end_constraint = end_kernel_constraint

    @end_bias_constraint.setter
    def end_bias_constraint(self, end_bias_constraint):
        self._end_bias_constraint = end_bias_constraint

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    @beta_1.setter
    def beta_1(self, beta_1):
        self._beta_1 = beta_1

    @beta_2.setter
    def beta_2(self, beta_2):
        self._beta_2 = beta_2

    @epsilon.setter
    def epsilon(self, epsilon):
        self._epsilon = epsilon

    @amsgrad.setter
    def amsgrad(self, amsgrad):
        self._amsgrad = amsgrad

    @clipnorm.setter
    def clipnorm(self, clipnorm):
        self._clipnorm = clipnorm

    @clipvalue.setter
    def clipvalue(self, clipvalue):
        self._clipvalue = clipvalue

    @global_clipnorm.setter
    def global_clipnorm(self, global_clipnorm):
        self._global_clipnorm = global_clipnorm
