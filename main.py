from config import Config
from functions.potential import get_potential
from functions.wave_functions import get_wave_packet
from plot.animate import animate
from tools.simulate import simulate, normalize
from wavefunctions.initializer import initialize_wavefunction


def main():
    config = Config()
    update_config = True
    if update_config:
        m = config.constants.mass
        p = config.initial_values.initial_momentum
        kinetic_energy = p**2 / (2*m)
        config.update("constants", potential_strength=1.2*kinetic_energy)
        config.update("initial_values", initial_position=-5)

    # TODO: NB same config must be loaded to all functions below. Should fix more stable solution soon
    # Define initial wave function and potential
    potential = get_potential(config=config, a=-10)

    # Initialize wave function object
    wave_function = initialize_wavefunction(config=config)
    wave_packet = wave_function.psi

    # Simulate time-evolution
    simulate_square_well = simulate(psi=wave_packet, v=potential, config=config)

    # Plot animation
    animate(simulation=simulate_square_well)
    return


if __name__ == '__main__':
    main()





