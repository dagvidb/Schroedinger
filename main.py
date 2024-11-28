"""
main.py
Executable script for solving the Schrödinger equation
"""
# New packages
from config import Config
from potentials.potential import Potential
from wavefunctions.initializer import initialize_wavefunction
from tools.simulate import hamiltonian
from plot.animate import complex_plot
import matplotlib.pyplot as plt
# Old packages
from plot.animate import animate
from tools.simulate import simulate, normalize
from simulations.simulation import Simulation


def main():
    config = Config(dimensions=1)
    update_config = True
    if update_config:
        m = config.constants.mass
        p = config.initial_values.initial_momentum
        kinetic_energy = p**2 / (2*m)
        config.update("constants", potential_strength=1.2*kinetic_energy)
        config.update("initial_values", initial_position=-5)

    # Initialize wave function object
    wave_function = initialize_wavefunction(config=config)
    # Define initial wave function and potential
    potential = Potential(config=config, a=-10)

    # Create and run simulation
    simulation = Simulation(wave_function, potential, config)
    simulation.run()

    results = simulation.get_results()

    # simulate_square_well = simulate(psi=wave_packet, v=potential, config=config)

    # Plot animation
    animate(simulation=results)
    return


if __name__ == '__main__':
    main()





