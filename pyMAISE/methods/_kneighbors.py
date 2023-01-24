from sklearn.neighbors import KNeighborsRegressor


class KNeighborsRegression:
    def __init__(self, parameters: dict = None):
        # Model Parameters
        self._n_neighbors = 5
        self._weights = "uniform"
        self._algorithm = "auto"
        self._leaf_size = 30
        self._p = 2
        self._metric = "minkowski"
        self._metric_params = None
        self._n_jobs = None

        # Change if user provided changes in dictionary
        if parameters != None:
            for key, value in parameters.items():
                setattr(self, key, value)

    # ===========================================================
    # Methods
    def regressor(self):
        return KNeighborsRegressor(
            n_neighbors=self._n_neighbors,
            weights=self._weights,
            algorithm=self._algorithm,
            leaf_size=self._leaf_size,
            p=self._p,
            metric=self._metric,
            metric_params=self._metric_params,
            n_jobs=self._n_jobs,
        )

    # ===========================================================
    # Getters
    @property
    def n_neighbors(self) -> int:
        return self._n_neighbors

    @property
    def weights(self):
        return self._weights

    @property
    def algorithm(self) -> str:
        return self._algorithm

    @property
    def leaf_size(self) -> int:
        return self._leaf_size

    @property
    def p(self) -> int:
        return self._p

    @property
    def metric(self):
        return self._metric

    @property
    def metric_params(self) -> dict:
        return self._metric_params

    @property
    def n_jobs(self) -> int:
        return self._n_jobs

    # ===========================================================
    # Setters
    @n_neighbors.setter
    def n_neighbors(self, n_neighbors: int):
        assert n_neighbors >= 1
        self._n_neighbors = n_neighbors

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @algorithm.setter
    def algorithm(self, algorithm: str):
        assert (
            algorithm == "auto"
            or algorithm == "ball_tree"
            or algorithm == "kd_tree"
            or algorithm == "brute"
        )
        self._algorithm = algorithm

    @leaf_size.setter
    def leaf_size(self, leaf_size: int):
        assert leaf_size > 0
        self._leaf_size = leaf_size

    @p.setter
    def p(self, p: int):
        assert p > 0
        self._p = p

    @metric.setter
    def metric(self, metric):
        self._metric = metric

    @metric_params.setter
    def metric_params(self, metric_params: dict):
        assert metric_params == None or isinstance(metric_params, dict)
        self._metric_params = metric_params

    @n_jobs.setter
    def n_jobs(self, n_jobs: int):
        assert n_jobs == None or n_jobs >= -1
        self._n_jobs = n_jobs
