import numpy as np
import math


def combined(time, initial_velocity, n, zeta, moment_inertia, c, magnetic_field):
    w_prime = (initial_velocity + n / (zeta + c * magnetic_field ** 2))
    d = n / (zeta + c * magnetic_field ** 2)
    return w_prime * np.exp(-(zeta + c * magnetic_field ** 2) * time / moment_inertia) - d


def constant(time, initial_velocity, n, moment_inertia):
    return initial_velocity - (n / moment_inertia) * time


def rmse(y_actual, y_predicted):
    mse = np.square(np.subtract(y_actual, y_predicted)).mean()
    return math.sqrt(mse)

