from config import InitialValues

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


initial_values = InitialValues()


def complex_plot(y, plot_prob=True, **kwargs):
    x = initial_values.x_grid
    real = np.real(y)
    imag = np.imag(y)
    a, *_ = plt.plot(x, real, label='Re', **kwargs)
    b,  *_ = plt.plot(x, imag, label='Im', **kwargs)
    plt.xlim(min(x), max(x))
    plt.legend()
    if plot_prob:

        abs_y = np.abs(y)
        prob_density = abs_y

        # Calculate mean position <x>
        mean_pos = np.sum(x * prob_density) / np.sum(prob_density)
        # Find closest x index to mean position
        x_loc = np.argmin(np.abs(x - mean_pos))

        p, *_ = plt.plot(x, abs_y, label='$ \sqrt{P}$')
        # q = plt.scatter(x[x_loc], 0, label='mean particle position')
        plt.legend()

        return a, b, p # , q

    else:
        return a, b


def animate(simulation, init_func=None):
    x = initial_values.x_grid
    fig = plt.figure()
    re, im, prob = complex_plot(y=simulation[0])
    plt.xlim(min(x)-1, max(x)+1)
    plt.ylim(-2, 2)
    plt.axvspan(5.0, 5.5, alpha=0.2, color='orange')

    if init_func:
        init_func()
        plt.legend()

    def update_frame(frame):
        y = simulation[frame]

        re.set_data((x, np.real(y)))
        im.set_data((x, np.imag(y)))
        # particle.set_data((x[x_loc], 2))
        # Update probability density and mean position

        abs_y = np.abs(y)
        prob_density = abs_y
        mean_pos = np.sum(x * prob_density) / np.sum(prob_density)
        x_loc = np.argmin(np.abs(x - mean_pos))
        prob.set_data(x, abs_y)
        # mean_pt.set_offsets([[x[x_loc], 0]])

        return prob, re, im #, mean_pt

    anim = FuncAnimation(fig, update_frame, frames=int(len(simulation)), interval=50)
    plt.show()
    # plt.close()

    return anim


def get_square_well():

    return
