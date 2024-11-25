import numpy as np
from tqdm import tqdm
from config import default_config


def normalize(f,config=default_config):
    dx = config.initial_values.dx
    norm = np.sum(np.square(np.abs(f))) * dx
    return f / np.sqrt(norm)


def d_dx2(f, config=default_config):
    dx = config.initial_values.dx
    df_dx2 = -2*f
    df_dx2[1:] += f[:-1]
    df_dx2[:-1] += f[1:]
    return df_dx2/(dx**2)


def hamiltonian(phi, v=0, config=default_config):
    m = config.constants.mass
    h = config.constants.h_bar ** 2
    return 1j*h/2/m * d_dx2(phi) - 1j*v*phi/h


def rk4(phi, dt, v=0):
    k1 = hamiltonian(phi, v=v)
    k2 = hamiltonian(phi + dt / 2 * k1, v=v)
    k3 = hamiltonian(phi + dt / 2 * k2, v=v)
    k4 = hamiltonian(phi + dt * k3, v=v)
    return phi + dt/6*(k1+2*k2+2*k3+k4)


def simulate(psi, v=0, config=default_config):

    dt = config.initial_values.dt
    steps = config.simulation_params.steps
    save_every = config.simulation_params.save_every

    simulation_steps = [np.copy(psi)]
    psi = normalize(psi)
    for i in tqdm(range(steps)):

        psi = rk4(psi, dt, v=v)
        psi = normalize(psi)

        if (i + 1) % save_every == 0:
            simulation_steps.append(np.copy(psi))

    return simulation_steps
