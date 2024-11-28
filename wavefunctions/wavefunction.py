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

    def apply_hamiltonian(self, psi: np.ndarray, potential: np.ndarray):
        """
        Apply the Hamiltonian operator for arbitrary dimensions.

        Args:
            psi (np.ndarray): The wave function to operate on.
            potential (np.ndarray): Potential array (same shape as `psi`).
        Returns:
            np.ndarray: Result of applying the Hamiltonian operator.
        """
        h_bar = self.config.constants.h_bar
        mass = self.config.constants.mass

        # Compute Laplacian (second derivatives) for N-dimensional grids
        laplacian = -2 * psi
        axes = tuple(range(psi.ndim))  # Operate over all dimensions
        for axis in axes:
            laplacian += np.roll(psi, shift=-1, axis=axis)
            laplacian += np.roll(psi, shift=1, axis=axis)
        laplacian /= self.grid_spacing**2

        # Hamiltonian: Kinetic + Potential
        kinetic_term = 1j * h_bar / (2 * mass) * laplacian
        potential_term = -1j * potential * psi / h_bar
        return kinetic_term + potential_term

    def time_evolve(self, potential: np.ndarray, method="RK4"):
        """
        Perform a single time step evolution using the RK4 method.

        Args:
            potential (np.ndarray): Potential array.
            method (str): Evolution method (currently only supports RK4).
        """
        if method == "RK4":
            k1 = self.apply_hamiltonian(self.psi, potential)
            k2 = self.apply_hamiltonian(self.psi + 0.5 * self.time_step * k1, potential)
            k3 = self.apply_hamiltonian(self.psi + 0.5 * self.time_step * k2, potential)
            k4 = self.apply_hamiltonian(self.psi + self.time_step * k3, potential)
            self.psi += self.time_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        self.normalize()
        self.current_time += self.time_step