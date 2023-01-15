import pyMAISE as mai
import numpy as np
from scipy.stats import uniform, randint


settings = {
    "verbosity": 1,
    "random_state": None,
    "test_size": 0.3,
}

model_settings = {
    "models": ["lasso", "dtree", "rforest"],
    # Lasso Regression
    "lasso": {"fit_intercept": True},
    # Decision Trees Regression
    "dtree": {"criterion": "squared_error", "splitter": "best"},
    # Random Forests Regression
    "rforest": {"criterion": "squared_error"},
}

param_grids = {
    "lasso": {"alpha": np.linspace(0.00001, 0.001, 5).tolist()},
    "dtree": {
        "max_depth": np.linspace(1, 200, 5, dtype=int).tolist(),
        "min_samples_split": np.linspace(2, 40, 5, dtype=int).tolist(),
        "min_samples_leaf": np.linspace(1, 15, 2, dtype=int).tolist(),
    },
    "rforest": {
        "n_estimators": np.linspace(50, 300, 5, dtype=int).tolist(),
        "max_depth": np.linspace(1, 200, 5, dtype=int).tolist(),
    },
}

param_distributions = {
    "lasso": {"alpha": uniform(loc=0.00001, scale=0.00099)},
    "dtree": {
        "max_depth": randint(low=1, high=200),
        "min_samples_split": randint(low=2, high=40),
        "min_samples_leaf": randint(low=1, high=15),
    },
    "rforest": {
        "n_estimators": randint(low=50, high=300),
        "max_depth": randint(low=1, high=200),
    },
}

# Initialize settings
globals = mai.settings.init(settings_changes=settings)

# Load cross section data from reactor physics data
# Equivalent to:
# preprocessor = mai.PreProcesser("path/to/xs.csv", slice(0, 8), slice(9))
preprocessor = mai.load_xs()

# Scale x data and get tuple of xtrain, xtest, ytrain, ytest
data = preprocessor.min_max_scale(scale_y=False)

# Hyper-parameter tuning object
tuning = mai.Tuning(data=data, model_settings=model_settings)

# Grid Search
grid = tuning.grid_search(param_grids=param_grids)
for key, value in grid.items():
    print(key)
    print(value)

# Random Search
random = tuning.random_search(param_distributions=param_distributions, n_iter=10)
for key, value in random.items():
    print(key)
    print(value)

# Bayesian Search
bayesian = tuning.bayesian_search(search_spaces=param_grids, n_iter=10)
for key, value in bayesian.items():
    print(key)
    print(value)
