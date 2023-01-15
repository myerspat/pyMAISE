import pyMAISE.settings as settings

from sklearn.ensemble import RandomForestRegressor


class RandomForestRegression:
    def __init__(self, parameters: dict = None):
        # Model parameters
        self._n_estimators = 100
        self._criterion = "squared_error"
        self._max_depth = None
        self._min_samples_split = 2
        self._min_samples_leaf = 1
        self._min_weight_fraction_leaf = 0.0
        self._max_features = None
        self._max_leaf_nodes = None
        self._min_impurity_decrease = 0.0
        self._bootstrap = True
        self._oob_score = False
        self._n_jobs = None
        self._warm_start = False
        self._ccp_alpha = 0.0
        self._max_samples = None

        # Change if user provided changes in dictionary
        if parameters != None:
            for key, value in parameters.items():
                setattr(self, key, value)

    # ===========================================================
    # Methods
    def regressor(self):
        return RandomForestRegressor(
            n_estimators=self._n_estimators,
            criterion=self._criterion,
            max_depth=self._max_depth,
            min_samples_split=self._min_samples_split,
            min_samples_leaf=self._min_samples_leaf,
            min_weight_fraction_leaf=self._min_weight_fraction_leaf,
            max_features=self._max_features,
            random_state=settings.values.random_state,
            max_leaf_nodes=self._max_leaf_nodes,
            min_impurity_decrease=self._min_impurity_decrease,
            bootstrap=self._bootstrap,
            oob_score=self._oob_score,
            verbose=settings.values.verbosity,
            warm_start=self._warm_start,
            ccp_alpha=self._ccp_alpha,
            max_samples=self._max_samples,
            n_jobs=self._n_jobs,
        )

    # ===========================================================
    # Getters
    @property
    def criterion(self) -> str:
        return self._criterion

    @property
    def max_depth(self) -> int:
        return self.max_depth

    @property
    def min_samples_split(self) -> float:
        return self._min_samples_split

    @property
    def min_samples_leaf(self) -> float:
        return self._min_samples_leaf

    @property
    def min_weight_fraction_leaf(self) -> float:
        return self._min_weight_fraction_leaf

    @property
    def max_features(self):
        return self._max_features

    @property
    def max_leaf_nodes(self) -> int:
        return self._max_leaf_nodes

    @property
    def min_impurity_decrease(self) -> float:
        return self._min_impurity_decrease

    @property
    def bootstrap(self) -> bool:
        return self._bootstrap

    @property
    def oob_score(self) -> bool:
        return self._oob_score

    @property
    def n_jobs(self) -> int:
        return self._n_jobs

    @property
    def warm_start(self) -> bool:
        return self._warm_start

    @property
    def ccp_alpha(self) -> float:
        return self._ccp_alpha

    @property
    def max_samples(self) -> float:
        return self._max_samples

    # ===========================================================
    # Setters
    @criterion.setter
    def criterion(self, criterion: str):
        assert (
            criterion == "squared_error"
            or criterion == "friedman_mse"
            or criterion == "absolute_error"
            or criterion == "poisson"
        )
        self._criterion = criterion

    @max_depth.setter
    def max_depth(self, max_depth: int):
        assert max_depth > 1 or max_depth == None
        self._max_depth = max_depth

    @min_samples_split.setter
    def min_samples_split(self, min_samples_split: float):
        assert min_samples_split >= 2
        self._min_samples_split = min_samples_split

    @min_samples_leaf.setter
    def min_samples_leaf(self, min_samples_leaf: float):
        assert min_samples_leaf >= 1
        self._min_samples_leaf = min_samples_leaf

    @min_weight_fraction_leaf.setter
    def min_weight_fraction_leaf(self, min_weight_fraction_leaf: float):
        assert min_weight_fraction_leaf >= 0
        self._min_weight_fraction_leaf = min_weight_fraction_leaf

    @max_features.setter
    def max_features(self, max_features):
        self._max_features = max_features

    @max_leaf_nodes.setter
    def max_leaf_nodes(self, max_leaf_nodes: int):
        assert max_leaf_nodes >= 0 or max_leaf_nodes == None
        self._max_leaf_nodes = max_leaf_nodes

    @min_impurity_decrease.setter
    def min_impurity_decrease(self, min_impurity_decrease: float):
        self._min_impurity_decrease = min_impurity_decrease

    @bootstrap.setter
    def bootstrap(self, bootstrap: bool):
        self._bootstrap = bootstrap

    @oob_score.setter
    def oob_score(self, oob_score: bool):
        self._oob_score = oob_score

    @n_jobs.setter
    def n_jobs(self, n_jobs: int):
        assert n_jobs > 0 or n_jobs == None
        self._n_jobs = n_jobs

    @warm_start.setter
    def warm_start(self, warm_start: bool):
        self._warm_start = warm_start

    @ccp_alpha.setter
    def ccp_alpha(self, ccp_alpha: float):
        assert ccp_alpha >= 0
        self._ccp_alpha = ccp_alpha

    @max_samples.setter
    def max_samples(self, max_samples: float):
        assert max_samples > 0 or max_samples == None
        self._max_samples = max_samples
