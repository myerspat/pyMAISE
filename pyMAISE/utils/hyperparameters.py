class HyperParameters:
    def __init__(self, default=None, parent_name=None, parent_values=None):
        self._default = default
        self._parent_name = parent_name
        self._parent_values = parent_values


class Boolean(HyperParameters):
    """
    Define a boolean hyperparameter. This is used in neural network hyperparameter
    tuning.

    Refer to
    `KerasTuner's documentation <https://keras.io/api/keras_tuner/hyperparameters/>`_
    for information on the arguments :cite:`chollet2015keras`.
    """

    def __init__(self, default=None, parent_name=None, parent_values=None):
        HyperParameters.__init__(self, default, parent_name, parent_values)

    # ===========================================================
    # Methods
    def hp(self, hp, hp_name):
        """
        Create an instance of ``keras_tuner.HyperParameters.Boolean``.

        Parameters
        ----------
        hp: keras_tuner.HyperParameters
            Base hyperparameter class.
        hp_name: str
            Name of the hyperparameter.

        Returns
        -------
        boolean_hp: keras_tuner.HyperParameters.Boolean
            Boolean KerasTuner hyperparameter.
        """
        return hp.Boolean(
            name=hp_name,
            default=self._default,
            parent_name=self._parent_name,
            parent_values=self._parent_values,
        )


class Int(HyperParameters):
    """
    Define an integer hyperparameter. This is used in neural network hyperparameter
    tuning.

    Refer to
    `KerasTuner's documentation <https://keras.io/api/keras_tuner/hyperparameters/>`_
    for information on the arguments :cite:`chollet2015keras`.
    """

    def __init__(
        self,
        min_value,
        max_value,
        step=None,
        sampling="linear",
        default=None,
        parent_name=None,
        parent_values=None,
    ):
        self._min_value = min_value
        self._max_value = max_value
        self._step = step
        self._sampling = sampling

        HyperParameters.__init__(self, default, parent_name, parent_values)

    # ===========================================================
    # Methods
    def hp(self, hp, hp_name):
        """
        Create an instance of ``keras_tuner.HyperParameters.Int``.

        Parameters
        ----------
        hp: keras_tuner.HyperParameters
            Base hyperparameter class.
        hp_name: str
            Name of the hyperparameter.

        Returns
        -------
        int_hp: keras_tuner.HyperParameters.Int
            Integer KerasTuner hyperparameter.
        """
        return hp.Int(
            name=hp_name,
            min_value=self._min_value,
            max_value=self._max_value,
            step=self._step,
            sampling=self._sampling,
            default=self._default,
            parent_name=self._parent_name,
            parent_values=self._parent_values,
        )


class Float(HyperParameters):
    """
    Define an floating point hyperparameter. This is used in neural network hyperparameter
    tuning.

    Refer to
    `KerasTuner's documentation <https://keras.io/api/keras_tuner/hyperparameters/>`_
    for information on the arguments :cite:`chollet2015keras`.
    """

    def __init__(
        self,
        min_value,
        max_value,
        step=None,
        sampling="linear",
        default=None,
        parent_name=None,
        parent_values=None,
    ):
        self._min_value = min_value
        self._max_value = max_value
        self._step = step
        self._sampling = sampling

        HyperParameters.__init__(self, default, parent_name, parent_values)

    # ===========================================================
    # Methods
    def hp(self, hp, hp_name):
        """
        Create an instance of ``keras_tuner.HyperParameters.Float``.

        Parameters
        ----------
        hp: keras_tuner.HyperParameters
            Base hyperparameter class.
        hp_name: str
            Name of the hyperparameter.

        Returns
        -------
        float_hp: keras_tuner.HyperParameters.Float
            Float KerasTuner hyperparameter.
        """
        return hp.Float(
            name=hp_name,
            min_value=self._min_value,
            max_value=self._max_value,
            step=self._step,
            sampling=self._sampling,
            default=self._default,
            parent_name=self._parent_name,
            parent_values=self._parent_values,
        )


class Choice(HyperParameters):
    """
    Define choice hyperparameter. This is used in neural network hyperparameter
    tuning. This can be used for string or other parameters where a range is not
    applicable.

    Refer to
    `KerasTuner's documentation <https://keras.io/api/keras_tuner/hyperparameters/>`_
    for information on the arguments :cite:`chollet2015keras`.
    """

    def __init__(
        self, values, ordered=None, default=None, parent_name=None, parent_values=None
    ):
        self._values = values
        self._ordered = ordered

        HyperParameters.__init__(self, default, parent_name, parent_values)

    # ===========================================================
    # Methods
    def hp(self, hp, hp_name):
        """
        Create an instance of ``keras_tuner.HyperParameters.Choice``.

        Parameters
        ----------
        hp: keras_tuner.HyperParameters
            Base hyperparameter class.
        hp_name: str
            Name of the hyperparameter.

        Returns
        -------
        choice_hp: keras_tuner.HyperParameters.Choice
            Choice KerasTuner hyperparameter.
        """
        return hp.Choice(
            name=hp_name,
            values=self._values,
            ordered=self._ordered,
            default=self._default,
            parent_name=self._parent_name,
            parent_values=self._parent_values,
        )

    # ===========================================================
    # Getters
    @property
    def values(self):
        """
        : list of int, float, str, or bool: The possible choices for the hyperparameter.
        """
        return self._values


class Fixed(HyperParameters):
    """
    Define fixed hyperparameter. This is used in neural network hyperparameter
    tuning.

    Refer to
    `KerasTuner's documentation <https://keras.io/api/keras_tuner/hyperparameters/>`_
    for information on the arguments :cite:`chollet2015keras`.
    """

    def __init__(self, value, parent_name=None, parent_values=None):
        self._value = value

        HyperParameters.__init__(
            self, parent_name=parent_name, parent_values=parent_values
        )

    # ===========================================================
    # Methods
    def hp(self, hp, hp_name):
        """
        Create an instance of ``keras_tuner.HyperParameters.Fixed``.

        Parameters
        ----------
        hp: keras_tuner.HyperParameters
            Base hyperparameter class.
        hp_name: str
            Name of the hyperparameter.

        Returns
        -------
        fixed_hp: keras_tuner.HyperParameters.Fixed
            Fixed KerasTuner hyperparameter.
        """
        return hp.Fixes(
            name=hp_name,
            value=self._value,
            parent_name=self._parent_name,
            parent_values=self._parent_values,
        )
