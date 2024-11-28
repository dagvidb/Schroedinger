import numpy as np
from config import Config


class Potential:
    """
    Class to represent and compute potential fields for simulations.

    The potential type and parameters are fetched directly from the Config object.
    """

    def __init__(self, config: Config, **kwargs):
        """
        Initialize the potential field.

        Args:
            config (Config): Configuration object containing grid, potential type, and parameters.
            kwargs: Additional parameters for the potential, such as:
                - "a", "b": Boundaries for square well/wall.
                - "v0": Strength of the potential.
                - "omega": Frequency for harmonic potential.
                - "n": Exponent for power potential.
        """
        self.config = config
        self.grid = self._get_grid_from_config(config)
        self.type = config.constants.potential_type  # Fetch potential type from Config
        self.strength = config.constants.potential_strength  # Fetch potential strength (v0) from Config
        self.params = kwargs

    def _get_grid_from_config(self, config: Config):
        """
        Extract the spatial grid from the configuration based on dimensions.

        Args:
            config (Config): The configuration object.

        Returns:
            np.ndarray or tuple: The spatial grid (1D, 2D, or 3D).
        """
        if config.initial_values.dimensions == 1:
            return config.initial_values.x_grid
        elif config.initial_values.dimensions == 2:
            return (config.initial_values.x_grid, config.initial_values.y_grid)
        elif config.initial_values.dimensions == 3:
            return (config.initial_values.x_grid, config.initial_values.y_grid, config.initial_values.z_grid)
        else:
            raise ValueError("Unsupported number of dimensions in config.")

    def get_potential(self) -> np.ndarray:
        """
        Compute the potential values over the grid.

        Returns:
            np.ndarray: The potential field corresponding to the selected type.
        """
        if self.type == "square_well":
            return self._square_well()
        elif self.type == "square_wall":
            return self._square_wall()
        elif self.type == "harmonic":
            return self._harmonic()
        elif self.type == "power":
            return self._power()
        else:
            raise ValueError(f"Unknown potential type: {self.type}")

    def _square_well(self) -> np.ndarray:
        """
        Compute a square well potential.

        Returns:
            np.ndarray: The square well potential field.
        """
        a = self.params.get("a", -5)  # Default boundaries
        b = self.params.get("b", 5)
        v0 = self.strength  # Use the strength from Config

        if isinstance(self.grid, np.ndarray):  # 1D case
            return np.where((self.grid > a) & (self.grid < b), 0, v0)
        else:  # N-dimensional case (2D, 3D)
            mask = np.logical_and(self.grid[0] >= a, self.grid[0] <= b)  # Apply to x-axis
            return np.where(mask, 0, v0)

    def _square_wall(self) -> np.ndarray:
        """
        Compute a square wall potential.

        Returns:
            np.ndarray: The square wall potential field.
        """
        a = self.params.get("a", -5)
        b = self.params.get("b", 5)
        d = self.params.get("d", 0.5)  # Wall thickness
        v0 = self.strength

        if isinstance(self.grid, np.ndarray):  # 1D case
            mask = ((self.grid >= (a - d)) & (self.grid <= a)) | ((self.grid >= b) & (self.grid <= (b + d)))
            return np.where(mask, v0, 0)
        else:  # N-dimensional case (2D, 3D)
            mask = np.logical_or(
                np.logical_and(self.grid[0] >= (a - d), self.grid[0] <= a),
                np.logical_and(self.grid[0] >= b, self.grid[0] <= (b + d))
            )
            return np.where(mask, v0, 0)

    def _harmonic(self) -> np.ndarray:
        """
        Compute a harmonic oscillator potential.

        Returns:
            np.ndarray: The harmonic potential field.
        """
        omega = self.params.get("omega", 1.0)  # Use default frequency if not provided

        if isinstance(self.grid, np.ndarray):  # 1D case
            return 0.5 * omega**2 * self.grid**2
        else:  # N-dimensional case (2D, 3D)
            r2 = sum(g**2 for g in self.grid)
            return 0.5 * omega**2 * r2

    def _power(self) -> np.ndarray:
        """
        Compute a power-law potential.

        Returns:
            np.ndarray: The power-law potential field.
        """
        n = self.params.get("n", 2)  # Default to quadratic potential
        v0 = self.strength

        if isinstance(self.grid, np.ndarray):  # 1D case
            return v0 * np.abs(self.grid)**n
        else:  # N-dimensional case (2D, 3D)
            r = np.sqrt(sum(g**2 for g in self.grid))
            return v0 * r**n
