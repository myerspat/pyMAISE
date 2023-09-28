# Sklean ML models
from ._linear import LinearRegression
from ._lasso import LassoRegression
from ._logistic_regression import Logistic_Regression
from ._dtree import DecisionTree
from ._rforest import RandomForest
from ._kneighbors import KNeighbors
from ._svr import SVRegression

# Old neural network architecture
from ._nn import NeuralNetsRegression
from ._nn_wrapper import NeuralNetsWrapper

# New neural network architecture
from .nn._nn_hypermodel import nnHyperModel
from pyMAISE.methods.nn import *
