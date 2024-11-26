import numpy as np
from config import Config


class WaveFunction:
    def __init__(self, psi: np.ndarray, grid_spacing: float, time_step: float, config: Config):
        """
        Initialize a WaveFunction for arbitrary dimensions.

        Args:
            psi (np.ndarray): The initial wave function (N-dimensional array).
            grid_spacing (float): Uniform spacing of the grid.
            time_step (float): Time step for evolution.
            config (Config): Configuration object with constants and parameters.
        """
        self.psi = psi  # Wave function data (can be 1D, 2D, or 3D)
        self.grid_spacing = grid_spacing  # dx, dy, dz for uniform grids
        self.time_step = time_step  # dt
        self.config = config  # Configuration parameters
        self.current_time = 0  # Track evolution time


    def normalize(self):
        """
        Normalize the wave function
        """
        norm = np.sum(np.abs(self.psi) ** 2) * self.grid_spacing**self.psi.ndim
        self.psi /= np.sqrt(norm)
