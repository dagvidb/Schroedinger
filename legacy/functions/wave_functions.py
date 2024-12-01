import numpy as np
from config import default_config
import warnings

warnings.warn(
    "The module 'wave_functions' is deprecated and will be removed in a future version. "
    "Please use 'new_module' instead.",
    DeprecationWarning,
    stacklevel=2,
)



def get_wave_packet(config=default_config):
    initial_values = config.initial_values
    x = initial_values.x_grid
    x0 = initial_values.initial_position
    p0 = initial_values.initial_momentum
    sigma = initial_values.initial_position_uncertainty
    return np.exp(1j*p0*x)*np.exp(-np.square(x-x0)/(2 * sigma ** 2), dtype=complex)
