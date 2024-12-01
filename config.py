import numpy as np


class Constants:
    """
    Class to hold physical constants used in the simulation.
    """
    def __init__(self) -> None:
        self.mass = 10.0               # Mass of the particle
        self.h_bar = 1.0               # Reduced Planck's constant
        self.potential_strength = 25.0  # Default strength of the potential
        self.potential_type = "square_well"  # Default type of potential


class InitialValues:
    """
    Class to hold initial values for the simulation, including spatial grid and wavefunction parameters.
    Supports 1D, 2D, and 3D simulations.
    """
    def __init__(self, dimensions: int = 1) -> None:
        self.dimensions = dimensions

        # Grid setup based on dimensions
        if dimensions == 1:
            self.x_grid = np.linspace(-10, 10, 3000)
            self.dx = self.x_grid[1] - self.x_grid[0]
            self.dt = self.dx**2
        elif dimensions == 2:
            x = np.linspace(-10, 10, 100)
            y = np.linspace(-10, 10, 100)
            self.x_grid, self.y_grid = np.meshgrid(x, y)
            self.dx = x[1] - x[0]
            self.dt = self.dx**2
        elif dimensions == 3:
            x = np.linspace(-10, 10, 50)
            y = np.linspace(-10, 10, 50)
            z = np.linspace(-10, 10, 50)
            self.x_grid, self.y_grid, self.z_grid = np.meshgrid(x, y, z)
            self.dx = x[1] - x[0]
            self.dt = self.dx**2
        else:
            raise ValueError("Only 1D, 2D, and 3D grids are supported.")

        # Wavefunction parameters
        self.initial_position = 0.0                  # Initial position of the wave packet
        self.initial_position_uncertainty = 1.0      # Width of the Gaussian wave packet
        self.initial_momentum = 20.0                 # Momentum of the wave packet


class SimulationParams:
    """
    Class to hold simulation parameters.
    """
    def __init__(self) -> None:
        self.steps = 2000          # Number of time steps to simulate
        self.save_every = 200      # Save the wavefunction every N steps


class Config:
    """
    Main configuration class that groups constants, initial values, and simulation parameters.
    """
    def __init__(self, dimensions: int = 1) -> None:
        self.constants = Constants()
        self.initial_values = InitialValues(dimensions=dimensions)
        self.simulation_params = SimulationParams()

    def update(self, category: str, **kwargs) -> None:
        """
        Update parameters dynamically for a specific category.

        Args:
            category (str): The category to update (e.g., 'constants', 'initial_values', 'simulation_params').
            kwargs: Key-value pairs of parameters to update.
        """
        if hasattr(self, category):
            category_obj = getattr(self, category)
            for key, value in kwargs.items():
                if hasattr(category_obj, key):
                    setattr(category_obj, key, value)
                else:
                    raise AttributeError(f"'{category}' has no attribute '{key}'")
        else:
            raise AttributeError(f"Config has no category '{category}'")

default_config = Config()