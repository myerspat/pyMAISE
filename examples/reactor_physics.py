import pyMAISE as mai
import numpy as np

settings = {
    "verbosity": 2,
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

tuning_settings = {
    "lasso": {"alpha": np.linspace(0.00001, 0.001, 30).tolist()},
    "dtree": {
        "max_depth": np.linspace(1, 200, 20, dtype=int).tolist(),
        "min_samples_split": np.linspace(0, 40, 20, dtype=int).tolist(),
        "min_samples_leaf": np.linspace(0, 15, 5, dtype=int).tolist(),
    },
    "rforest": {
        "n_estimators": np.linspace(50, 300, 20, dtype=int).tolist(),
        "max_depth": np.linspace(1, 200, 20, dtype=int).tolist(),
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
grid = tuning.grid_search(tuning_settings=tuning_settings)

# Random Search
random = tuning.random_search(tuning_settings=tuning_settings)

# Bayesian Search
bayesian = tuning.bayesian_search(tuning_settings=tuning_settings)
