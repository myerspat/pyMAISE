import pyMAISE.settings as settings

from sklearn.linear_model import Lasso


class LassoRegression:
    def __init__(self, parameters: dict = None):
        # Model Parameters
        self._alpha = 1.0
        self._fit_intercept = True
        self._precompute = False
        self._copy_X = True
        self._max_iter = 1000
        self._tol = 1e-4
        self._warm_start = False
        self._positive = False
        self._selection = "cyclic"

        # Change if user provided changes in dictionary
        if parameters != None:
            for key, value in parameters.items():
                setattr(self, key, value)

    # ===========================================================
    # Methods
    def regressor(self):
        return Lasso(
            alpha=self._alpha,
            fit_intercept=self._fit_intercept,
            precompute=self._precompute,
            copy_X=self._copy_X,
            max_iter=self._max_iter,
            tol=self._tol,
            warm_start=self._warm_start,
            positive=self._positive,
            selection=self._selection,
            random_state=settings.values.random_state,
        )

    # ===========================================================
    # Getters
    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def fit_intercept(self) -> bool:
        return self._fit_intercept

    @property
    def precompute(self) -> bool:
        return self._precompute

    @property
    def copy_X(self) -> bool:
        return self._copy_X

    @property
    def max_iter(self) -> int:
        return self._max_iter

    @property
    def positive(self) -> bool:
        return self._positive

    @property
    def selection(self) -> str:
        return self._selection

    # ===========================================================
    # Setters
    @alpha.setter
    def alpha(self, alpha: float):
        assert alpha >= 0
        self._alpha = alpha

    @fit_intercept.setter
    def fit_intercept(self, fit_intercept: bool):
        self._fit_intercept = fit_intercept

    @precompute.setter
    def precompute(self, precompute: bool):
        self._precompute = precompute

    @copy_X.setter
    def copy_X(self, copy_X: bool):
        self._copy_X = copy_X

    @max_iter.setter
    def max_iter(self, max_iter: int):
        assert max_iter > 0
        self._max_iter = max_iter

    @positive.setter
    def positive(self, positive: bool):
        self._positive = positive

    @selection.setter
    def selection(self, selection: str):
        assert selection == "cyclic" or selection == "random"
        self._selection = selection
