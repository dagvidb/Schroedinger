import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting


class Visualization:
    """
    A class to handle visualization of wavefunction simulations.
    Supports 1D and 2D visualizations.
    """

    def __init__(self, config, results):
        """
        Initialize the Visualization object.

        Args:
            config: Configuration object containing grid and parameters.
            results (list of np.ndarray): Snapshots of the wavefunction over time.
        """
        self.config = config
        self.results = results
        self.dimensions = config.initial_values.dimensions

    def plot_static(self, time_indices=None):
        """
        Plot static wavefunction components (Re(ψ), Im(ψ), |ψ|^2) at specific time steps.

        Args:
            time_indices (list of int, optional): Indices of time steps to plot.
        """
        if time_indices is None:
            time_indices = [0, len(self.results) // 2, len(self.results) - 1]

        if self.dimensions == 1:
            x = self.config.initial_values.x_grid
            for i in time_indices:
                psi = self.results[i]
                plt.plot(x, np.real(psi), label=f"Re(ψ) at Step {i}")
                plt.plot(x, np.imag(psi), label=f"Im(ψ) at Step {i}")
                plt.plot(x, np.abs(psi)**2, label=f"|ψ|^2 at Step {i}")
            plt.xlabel("x")
            plt.ylabel("Wavefunction")
            plt.title("Wavefunction Components Over Time")
            plt.legend()
            plt.show()

        elif self.dimensions == 2:
            raise NotImplementedError("Animations for 3D are not supported yet.")
            x = self.config.initial_values.x_grid
            y = self.config.initial_values.y_grid
            X, Y = np.meshgrid(x, y)

            for i in time_indices:
                psi = self.results[i]

                # 3D Surface Plot for |ψ|^2
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(X, Y, np.abs(psi)**2, cmap='viridis', edgecolor='none')
                ax.set_title(f"|ψ|^2 at Time Step {i}")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("|ψ|^2")
                plt.show()

                # 3D Surface Plot for Re(ψ)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(X, Y, np.real(psi), cmap='coolwarm', edgecolor='none')
                ax.set_title(f"Re(ψ) at Time Step {i}")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("Re(ψ)")
                plt.show()

                # 3D Surface Plot for Im(ψ)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(X, Y, np.imag(psi), cmap='coolwarm', edgecolor='none')
                ax.set_title(f"Im(ψ) at Time Step {i}")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("Im(ψ)")
                plt.show()

        else:
            raise NotImplementedError("Static plots for 3D are not supported yet.")

    def animate(self):
        """
        Animate the probability density |ψ|^2 over time.

        Returns:
            FuncAnimation: The animation object.
        """
        if self.dimensions == 1:
            x = self.config.initial_values.x_grid
            fig, ax = plt.subplots()
            line, = ax.plot([], [], lw=2, label="|ψ|^2")
            re_line, = ax.plot([], [], lw=1, label="Re(ψ)")
            im_line, = ax.plot([], [], lw=1, label="Im(ψ)")
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(-1.1, 1.1)  # Adjust based on the expected range of wavefunction components
            ax.set_xlabel("x")
            ax.set_ylabel("Wavefunction Components")
            ax.set_title("Wavefunction Evolution")
            ax.legend()

            def init():
                line.set_data([], [])
                re_line.set_data([], [])
                im_line.set_data([], [])
                return line, re_line, im_line

            def update(frame):
                psi = self.results[frame]
                line.set_data(x, np.abs(psi)**2)
                re_line.set_data(x, np.real(psi))
                im_line.set_data(x, np.imag(psi))
                return line, re_line, im_line

            anim = FuncAnimation(fig, update, frames=len(self.results), init_func=init, blit=True, interval=50)
            plt.show()
            return anim

        elif self.dimensions == 2:
            raise NotImplementedError("Animations for 3D are not supported yet.")
            x = self.config.initial_values.x_grid
            y = self.config.initial_values.y_grid
            X, Y = np.meshgrid(x, y)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            surface = ax.plot_surface(X, Y, np.abs(self.results[0])**2, cmap='viridis')

            def update(frame):
                ax.clear()
                ax.plot_surface(X, Y, np.abs(self.results[frame])**2, cmap='viridis')
                ax.set_title(f"|ψ|^2 at Time Step {frame}")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("|ψ|^2")
                return surface,

            anim = FuncAnimation(fig, update, frames=len(self.results), interval=50)
            plt.show()
            return anim

        else:
            raise NotImplementedError("Animations for 3D are not supported yet.")
