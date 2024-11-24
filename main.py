from config import Config
from functions.potential import *
from functions.wave_functions import get_wave_packet
from plot.animate import animate
from tools.simulate import simulate

config = Config()


def main():
    update_config = True
    if update_config:
        m = config.constants.mass
        p = config.initial_values.initial_momentum
        kinetic_energy = p**2 / (2*m)
        config.update("constants", potential_strength=0.8*kinetic_energy)
        config.update("initial_values", initial_position=-8)

    # Define initial wave function and potential
    wave_packet = get_wave_packet()
    potential = get_square_wall_right()

    # Simulate time-evolution
    simulate_square_well = simulate(psi=wave_packet, v=potential)

    # Plot animation
    anim = animate(simulation=simulate_square_well)
    return


if __name__ == '__main__':
    main()





