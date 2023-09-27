import keras_tuner as kt


class HyperParameters:
    def __init__(self, default=None, parent_name=None, parent_values=None):
        self._default = default
        self._parent_name = parent_name
        self._parent_values = parent_values


class Boolean(HyperParameters):
    # ===========================================================
    # Methods
    def hp(self, hp, hp_name):
        return hp.Boolean(
            name=hp_name,
            default=self._default,
            parent_name=self._parent_name,
            parent_values=self._parent_values,
        )


class Int(HyperParameters):
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
    def __init__(
        self, values, ordered=None, default=None, parent_name=None, parent_values=None
    ):
        self._values = values
        self._ordered = ordered

        HyperParameters.__init__(self, default, parent_name, parent_values)

    # ===========================================================
    # Methods
    def hp(self, hp, hp_name):
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
        return self.values


class Fixed(HyperParameters):
    def __init__(self, value, parent_name=None, parent_values=None):
        self._value = value

        HyperParameters.__init__(
            self, parent_name=parent_name, parent_values=parent_values
        )

    # ===========================================================
    # Methods
    def hp(self, hp, hp_name):
        return hp.Fixes(
            name=hp_name,
            value=self._value,
            parent_name=self._parent_name,
            parent_values=self._parent_values,
        )
