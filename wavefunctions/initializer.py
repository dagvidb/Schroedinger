import numpy as np
from wavefunctions.wavefunction import WaveFunction
from config import Config

def initialize_wavefunction(config: Config, kind="gaussian", **kwargs):
    """
    Initialize a wave function based on the configuration and desired type.

    Args:
        config (Config): Configuration object containing grid and parameters.
        kind (str): Type of initial wavefunction. Options: 'gaussian', 'plane_wave', 'custom'.
        kwargs: Additional parameters for customization.

    Returns:
        WaveFunction: An initialized WaveFunction object.
    """
    if config.initial_values.dimensions == 1:
        x_grid = config.initial_values.x_grid

        if kind == "gaussian":
            x0 = kwargs.get("x0", config.initial_values.initial_position)
            sigma = kwargs.get("sigma", config.initial_values.initial_position_uncertainty)
            p0 = kwargs.get("p0", config.initial_values.initial_momentum)
            psi = np.exp(1j * p0 * x_grid) * np.exp(-((x_grid - x0)**2) / (2 * sigma**2), dtype=complex)

        elif kind == "plane_wave":
            p0 = kwargs.get("p0", config.initial_values.initial_momentum)
            psi = np.exp(1j * p0 * x_grid, dtype=complex)

        elif kind == "custom":
            custom_func = kwargs.get("custom_func", lambda x: np.zeros_like(x, dtype=complex))
            psi = custom_func(x_grid)

        else:
            raise ValueError(f"Unknown wavefunction type: {kind}")

    elif config.initial_values.dimensions == 2:
        x_grid, y_grid = config.initial_values.x_grid, config.initial_values.y_grid

        if kind == "gaussian":
            x0, y0 = kwargs.get("x0", 0), kwargs.get("y0", 0)
            sigma = kwargs.get("sigma", 1.0)
            p0x, p0y = kwargs.get("p0x", 0), kwargs.get("p0y", 0)
            psi = (np.exp(1j * (p0x * x_grid + p0y * y_grid)) *
                   np.exp(-((x_grid - x0)**2 + (y_grid - y0)**2) / (2 * sigma**2), dtype=complex))

        elif kind == "plane_wave":
            p0x, p0y = kwargs.get("p0x", 0), kwargs.get("p0y", 0)
            psi = np.exp(1j * (p0x * x_grid + p0y * y_grid), dtype=complex)

        elif kind == "custom":
            custom_func = kwargs.get("custom_func", lambda x, y: np.zeros_like(x, dtype=complex))
            psi = custom_func(x_grid, y_grid)

        else:
            raise ValueError(f"Unknown wavefunction type: {kind}")

    else:
        raise NotImplementedError(f"Wavefunction initialization not implemented for {config.initial_values.dimensions}D.")

    # Normalize and return a WaveFunction instance
    wavefunction = WaveFunction(psi=psi, grid_spacing=config.initial_values.dx,
                                time_step=config.initial_values.dt, config=config)
    wavefunction.normalize()
    return wavefunction
