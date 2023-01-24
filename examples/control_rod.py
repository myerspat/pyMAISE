import pyMAISE as mai
from scipy.stats import uniform, randint

# Global settings
settings = {
    "verbosity": 1,
    "random_state": None,
    "test_size": 0.3,
}

# Initialize pyMAISE
globals = mai.settings.init(settings_changes=settings)

# Load MIT reactor control rod data
preprocessor = mai.load_rod_positions()

# Scale data using minmax scaling
data = preprocessor.min_max_scale()

# Neural Network settings
model_settings = {
    "models": ["nn"],
    "nn": {
        # Sequential
        "num_layers": 4,
        "dropout": True,
        "rate": 0.5,
        "validation_split": 0.15,
        "loss": "mean_absolute_error",
        "metrics": "mean_absolute_error",
        "batch_size": 8,
        "epochs": 50,
        "warm_start": True,
        "jit_compile": False,
        # Starting Layer
        "start_num_nodes": 100,
        "start_kernel_initializer": "normal",
        "start_activation": "relu",
        "input_dim": 6,
        # Middle Layers
        "mid_num_node_strategy": "linear",
        "mid_kernel_initializer": "normal",
        "mid_activation": "relu",
        # Ending Layer
        "end_num_nodes": 22,
        "end_activation": "linear",
        "end_kernel_initializer": "normal",
        # Optimizer
        "optimizer": "adam",
        "learning_rate": 5e-4,
    },
}

# Random search parameter space
param_distributions = {
    "nn": {
        "batch_size": [8, 16, 32, 64],
        "learning_rate": uniform(loc=1e-5, scale=0.00099),
        "num_layers": [2, 3, 4, 5, 6],
        "start_num_nodes": randint(low=50, high=200),
    }
}

# Random search hyper-parameter tuning
tuning = mai.Tuning(data=data, model_settings=model_settings)
random = tuning.random_search(param_distributions=param_distributions, n_iter=5)
for key, value in random.items():
    print(key)
    print(value.iloc[:, 4:])
