from config import Constants, InitialValues
import numpy as np

constants = Constants()
initial_values = InitialValues()


def get_power_potential(n=2):
    x = initial_values.x_grid
    v0 = constants.potential_strength
    return v0*x**n


def get_square_well(a=-5, b=5):
    x = initial_values.x_grid
    v0 = constants.potential_strength
    return np.where((x > a) & (x < b), 0, v0)


def get_square_wall(a=-5, b=5, d=0.5):
    x = initial_values.x_grid
    v0 = constants.potential_strength
    return np.where(((a - d <= x) & (x <= a)) | ((b <= x) & (x <= b + d)), v0, 0)


def get_square_wall_right(b=5, d=0.5):
    x = initial_values.x_grid
    v0 = constants.potential_strength
    return np.where(((b <= x) & (x <= b + d)), v0, 0)


def get_square_wall_left(a=-5, d=0.5):
    x = initial_values.x_grid
    v0 = constants.potential_strength
    return np.where(((a - d <= x) & (x <= a)), v0, 0)


