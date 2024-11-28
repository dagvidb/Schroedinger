import numpy as np
from tqdm import tqdm
from wavefunctions.wavefunction import WaveFunction
from potentials.potential import Potential
from config import Config


class Simulation:
    """
    A class to manage and run simulations for wavefunction dynamics.

    Handles the time evolution of a wavefunction in a given potential using RK4.
    """

    def __init__(self, wavefunction: WaveFunction, potential: Potential, config: Config):
        """
        Initialize the simulation.

        Args:
            wavefunction (WaveFunction): The wavefunction to simulate.
            potential (Potential): The potential in which the wavefunction evolves.
            config (Config): The configuration object with simulation parameters.
        """
        self.wavefunction = wavefunction
        self.potential = potential
        self.config = config
        self.results = []  # Store wavefunction snapshots over time

    def run(self, steps: int = None, save_every: int = None):
        """
        Run the simulation over a number of steps.

        Args:
            steps (int): Number of time steps to simulate (default: from config).
            save_every (int): Save the wavefunction every N steps (default: from config).
        """
        # Default to configuration values
        steps = steps or self.config.simulation_params.steps
        save_every = save_every or self.config.simulation_params.save_every

        # Compute the potential field
        potential_field = self.potential.get_potential()

        # Normalize the wavefunction initially
        self.wavefunction.normalize()

        # Save the initial state
        self.results.append(self.wavefunction.psi.copy())

        # Time evolution loop
        for step in tqdm(range(steps), desc="Simulating"):
            self.wavefunction.time_evolve(potential_field)

            if (step + 1) % save_every == 0:
                self.results.append(self.wavefunction.psi.copy())

    def get_results(self):
        """
        Retrieve the simulation results.

        Returns:
            list of np.ndarray: Snapshots of the wavefunction at saved intervals.
        """
        return self.results

    def compute_observable(self, operator: np.ndarray) -> list:
        """
        Compute the expectation value of an observable over time.

        Args:
            operator (np.ndarray): The operator matrix (e.g., position or momentum operator).

        Returns:
            list: Expectation values at each saved step.
        """
        observables = []
        for psi in self.results:
            expectation_value = np.real(np.sum(np.conj(psi) * operator * psi) * self.wavefunction.grid_spacing)
            observables.append(expectation_value)
        return observables
