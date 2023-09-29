import pyMAISE.settings as settings

from sklearn.svm import SVR, SVC


class SVM:
    def __init__(self, parameters: dict = None):
        # Model parameters

        # Both classes share
        self._kernel='rbf'
        self._C=1.0
        self._degree=3
        self._gamma = "scale"
        self._coef0=0.0
        self._shrinking=True
        self._tol=0.001
        self._cache_size=200
        self._max_iter=-1
        self._verbose=settings.values.verbosity

        # Unique class variables
        if settings.values.regression:
        
            self._epsilon = 0.1
            self._shrinking = True

        if settings.values.classification:
            
            self._probability=False
            self._class_weight=None
            self._decision_function_shape='ovr'
            self._break_ties=False
            self._random_state=settings.values.random_state

        # Change if user provided changes in dictionary
        if parameters != None:
            for key, value in parameters.items():
                setattr(self, key, value)

    # ===========================================================
    # Methods
    def regressor(self):

        if settings.values.regression:
            return SVR(
                kernel=self._kernel,
                degree=self._degree,
                gamma=self._gamma,
                coef0=self._coef0,
                tol=self._tol,
                C=self._C,
                epsilon=self._epsilon,
                shrinking=self._shrinking,
                cache_size=self._cache_size,
                verbose=settings.values.verbosity,
                max_iter=self._max_iter,
            )
        if settings.values.classification:
            return SVC(
                kernel=self._kernel,
                degree=self._degree,
                gamma=self._gamma,
                coef0=self._coef0,
                tol=self._tol,
                C=self._C,
                cache_size=self._cache_size,
                verbose=settings.values.verbosity,
                max_iter=self._max_iter,

                probability=self._probability,
                class_weight=self._class_weight,
                decision_function_shape=self._decision_function_shape,
                break_ties=self._break_ties,
                random_state=self._random_state,

            )

    # ===========================================================
    # Getters
    @property
    def kernel(self):
        return self._kernel

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def gamma(self):
        return self.gamma

    @property
    def coef0(self) -> float:
        return self._coef0

    @property
    def tol(self) -> float:
        return self._tol

    @property
    def C(self) -> float:
        return self._C

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def shrinking(self) -> bool:
        return self._shrinking

    @property
    def cache_size(self) -> float:
        return self._cache_size

    @property
    def max_iter(self) -> int:
        return self._max_iter

    @property
    def probability(self):
        return self._probability
    @property
    def class_weight(self):
        return self._class_weight
    @property
    def decision_function_shape(self):
        return self._decision_function_shape
    @property
    def break_ties(self):
        return self._break_ties

    

    # ===========================================================
    # Setters
    @kernel.setter
    def kernel(self, kernel):
        self._kernel = kernel

    @degree.setter
    def degree(self, degree: int):
        assert degree >= 0
        self._degree = degree

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma

    @coef0.setter
    def coef0(self, coef0: float):
        self._coef0 = coef0

    @tol.setter
    def tol(self, tol: float):
        self._tol = tol

    @C.setter
    def C(self, C: float):
        assert C > 0
        self._C = C

    @epsilon.setter
    def epsilon(self, epsilon: float):
        assert epsilon > 0
        self._epsilon = epsilon

    @shrinking.setter
    def shrinking(self, shrinking: bool):
        self._shrinking = shrinking

    @cache_size.setter
    def cache_size(self, cache_size: float):
        self._cache_size = cache_size

    @max_iter.setter
    def max_iter(self, max_iter: int):
        assert max_iter >= -1
        self._max_iter = max_iter
    
    @probability.setter
    def probability(self, probability):
        self._probability = probability

    @class_weight.setter
    def class_weight(self, class_weight):
        self._class_weight = class_weight

    @decision_function_shape.setter
    def decision_function_shape(self, decision_function_shape):
        self._decision_function_shape = decision_function_shape

    @break_ties.setter
    def break_ties(self, break_ties):
        self._break_ties = break_ties

