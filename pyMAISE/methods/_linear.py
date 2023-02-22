from sklearn.linear_model import LinearRegression as LinearRegressor


class LinearRegression:
    def __init__(self, parameters: dict = None):
        # Model parameters
        self._fit_intercept = True
        self._copy_X = True
        self._n_jobs = None
        self._positive = False

        # Change if user provided changes in dictionary
        if parameters != None:
            for key, value in parameters.items():
                setattr(self, key, value)

    # ===========================================================
    # Methods
    def regressor(self):
        return LinearRegressor(
            fit_intercept=self._fit_intercept,
            copy_X=self._copy_X,
            n_jobs=self._n_jobs,
            positive=self._positive,
        )

    # ===========================================================
    # Getters
    @property
    def fit_intercept(self) -> bool:
        return self._fit_intercept

    @property
    def copy_X(self) -> bool:
        return self._copy_X

    @property
    def n_jobs(self) -> int:
        return self._n_jobs

    @property
    def positive(self) -> bool:
        return self._positive

    # ===========================================================
    # Setters
    @fit_intercept.setter
    def fit_intercept(self, fit_intercept: bool):
        self._fit_intercept = fit_intercept

    @copy_X.setter
    def copy_X(self, copy_X: bool):
        self._copy_X = copy_X

    @n_jobs.setter
    def n_jobs(self, n_jobs: int):
        assert n_jobs >= -1
        self.n_jobs = n_jobs

    @positive.setter
    def positive(self, positive: bool):
        self.positive = positive
