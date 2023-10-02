from sklearn.linear_model import LogisticRegression

import pyMAISE.settings as settings


class Logistic_Regression:
    def __init__(self, parameters: dict = None):
        # Model Parameters
        self._penalty = "l2"
        self._dual = False
        self._tol = 0.0001
        self._C = 1.0
        self._fit_intercept = True
        self._intercept_scaling = 1
        self._class_weight = None
        self._solver = "lbfgs"
        self._max_iter = 100
        self._multi_class = "auto"
        self._verbose = 0
        self._warm_start = False
        self._n_jobs = None
        self._l1_ratio = None

        # Change if user provided changes in dictionary
        if parameters != None:
            for key, value in parameters.items():
                setattr(self, key, value)

    # =======================================================
    #  Methods
    def regressor(self):
        return LogisticRegression(
            penalty=self._penalty,
            dual=self._dual,
            tol=self._tol,
            C=self._C,
            fit_intercept=self._fit_intercept,
            intercept_scaling=self._intercept_scaling,
            class_weight=self._class_weight,
            random_state=settings.values.random_state,
            solver=self._solver,
            max_iter=self._max_iter,
            multi_class=self._multi_class,
            verbose=self._verbose,
            warm_start=self._warm_start,
            n_jobs=self._n_jobs,
            l1_ratio=self._l1_ratio,
        )

    # ========================================================
    # Getters
    @property
    def penalty(self):
        return self._penalty

    @property
    def dual(self):
        return self._dual

    @property
    def tol(self):
        return self._tol

    @property
    def C(self):
        return self._C

    @property
    def fit_intercept(self):
        return self._fit_intercept

    @property
    def intercept_scaling(self):
        return self._intercept_scaling

    @property
    def class_weight(self):
        return self._class_weight

    @property
    def solver(self):
        return self._solver

    @property
    def max_iter(self):
        return self._max_iter

    @property
    def multi_class(self):
        return self._multiclass

    @property
    def verbose(self):
        return self._verbose

    @property
    def warm_start(self):
        return self._warm_start

    @property
    def n_jobs(self):
        return self._n_jobs

    @property
    def l1_ratio(self):
        return self._l1_ratio

    # Setters
    @penalty.setter
    def penalty(self, penalty):
        self._penalty = penalty

    @dual.setter
    def dual(self, dual):
        self._dual = dual

    @tol.setter
    def tol(self, tol):
        self._tol = tol

    @C.setter
    def C(self, C):
        self._C = C

    @fit_intercept.setter
    def fit_intercept(self, fit_intercept):
        self._fit_intercept = fit_intercept

    @intercept_scaling.setter
    def intercept_scaling(self, intercept_scaling):
        self._intercept_scaling = intercept_scaling

    @class_weight.setter
    def class_weight(self, class_weight):
        self._class_weight = class_weight

    @solver.setter
    def solver(self, solver):
        self._solver = solver

    @max_iter.setter
    def max_iter(self, max_iter):
        self._max_iter = max_iter

    @multi_class.setter
    def multi_class(self, multi_class):
        self._multiclass = multi_class

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose

    @warm_start.setter
    def warm_start(self, warm_start):
        self._warm_start = warm_start

    @n_jobs.setter
    def n_jobs(self, n_jobs):
        self._n_jobs = n_jobs

    @l1_ratio.setter
    def l1_ratio(self, l1_ratio):
        self._l1_ratio = l1_ratio
