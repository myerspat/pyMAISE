import pyMAISE.settings as settings

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

class DecisionTree:
    def __init__(self, parameters: dict = None):
        # Model parameters

        if settings.values.regression:
            self._criterion = "squared_error"
        else:
            self._criterion = "gini"
            self._class_weight = None

        self._splitter = "best"
        self._max_depth = None
        self._min_samples_split = 2
        self._min_samples_leaf = 1
        self._min_weight_fraction_leaf = 0.0
        self._max_features = None
        self._max_leaf_nodes = None
        self._min_impurity_decrease = 0.0
        self._ccp_alpha = 0.0

        # Change if user provided changes in dictionary
        if parameters != None:
            for key, value in parameters.items():
                setattr(self, key, value)

    # ===========================================================
    # Methods
    def regressor(self):

        if settings.values.regression:
            return DecisionTreeRegressor(
                criterion=self._criterion,
                splitter=self._splitter,
                max_depth=self._max_depth,
                min_samples_split=self._min_samples_split,
                min_samples_leaf=self._min_samples_leaf,
                min_weight_fraction_leaf=self._min_weight_fraction_leaf,
                max_features=self._max_features,
                random_state=settings.values.random_state,
                max_leaf_nodes=self._max_leaf_nodes,
                min_impurity_decrease=self._min_impurity_decrease,
                ccp_alpha=self._ccp_alpha,
            )
        else:
            return DecisionTreeClassifier(
                    criterion=self._criterion,
                    splitter=self._splitter,
                    max_depth=self._max_depth,
                    min_samples_split = self._min_samples_split,
                    min_samples_leaf = self._min_samples_leaf,
                    min_weight_fraction_leaf=self._min_weight_fraction_leaf,
                    max_features=self._max_features,
                    random_state=settings.values.random_state,
                    max_leaf_nodes=self._max_leaf_nodes,
                    min_impurity_decrease=self._min_impurity_decrease,
                    class_weight=self._class_weight,
                    ccp_alpha=self._ccp_alpha,

                    )

    # ===========================================================
    # Getters
    @property
    def criterion(self) -> str:
        return self._criterion

    @property
    def splitter(self) -> str:
        return self._splitter

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
    def ccp_alpha(self) -> float:
        return self._ccp_alpha

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

    @splitter.setter
    def splitter(self, splitter: str):
        assert splitter == "best" or splitter == "random"
        self._splitter = splitter

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

    @ccp_alpha.setter
    def ccp_alpha(self, ccp_alpha: float):
        assert ccp_alpha >= 0
        self._ccp_alpha = ccp_alpha
