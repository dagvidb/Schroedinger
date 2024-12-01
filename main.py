"""
main.py
Executable script for solving the Schrödinger equation
"""
# New packages
from config import Config
from potentials.potential import Potential
from wavefunctions.initializer import initialize_wavefunction
from simulations.simulation import Simulation
from visualization.visualizer import Visualization
# Old packages
from legacy.plot import animate
from legacy.tools.simulate import simulate
from legacy.functions.wave_functions import get_wave_packet
from legacy.functions.potential import get_potential



def main():
    config: Config = Config(dimensions=2)
    update_config: bool = True

    if update_config:
        m: float = config.constants.mass
        p: float = config.initial_values.initial_momentum
        kinetic_energy: float = p**2 / (2*m)
        config.update("constants", potential_strength=1.2*kinetic_energy)
        config.update("initial_values", initial_position=-5)

    use_legacy: bool = False
    if not use_legacy:

        wave_function = initialize_wavefunction(config=config)
        potential = Potential(config=config, a=-10)
        simulation = Simulation(wave_function, potential, config)
        simulation.run()
        results = simulation.get_results()
        viz = Visualization(config=config, results=results)
        viz.animate()

    else:

        wave_function = get_wave_packet(config)
        potential = get_potential(config=config, a=-10)
        results = simulate(psi=wave_function, v=potential, config=config)
        animate(simulation=results)

    return


if __name__ == '__main__':
    main()





