"""Design a data-driven controller for a pendulum via the Lie derivative."""

import numpy as np
import scipy.integrate
import SumOfSquares
import sympy
from matplotlib import pyplot as plt


def main():
    """Design a data-driven controller for a pendulum."""
    # Set random seed
    rng = np.random.default_rng(1234)

    # Max degree of x1 and x3 in phi dictionary of obserables
    max_phi = 3
    # Max degree of x1 and x3 in psi dictionary of obserables
    max_psi = 4
    # Degree of the controller u as a function of x variables
    deg_u = 4
    # Decay parameter for Lyapunov function
    alpha = 100
    # State space parameter
    eta = np.sqrt(0.95)

    # Timestep (s)
    dt = 0.01
    # Time array
    t_span = (0, 20)
    t = np.arange(*t_span, dt)
    # Number of initial conditions
    num_ics = 20

    def sinusoidal_forcing(t, a, b):
        """Compute sinusodial forcing input for pendulum."""
        u = a * np.sin(t + b)
        return u

    def pendulum_ivp(t, x, a, b):
        """Compute state derivative of pendulum with sinusoidal forcing."""
        u = sinusoidal_forcing(t, a, b)
        x_dot = np.array([
            x[1],
            np.sin(x[0]) - 0.1 * x[1] - np.cos(x[0]) * u,
        ])
        return x_dot

    X_lst = []
    Y_lst = []
    U_lst = []
    for i in range(num_ics):
        # Randomized initial conditions and inputs
        x0_i = rng.uniform(
            low=np.array([0, -2]),
            high=np.array([2 * np.pi, 2]),
        )
        a_i = rng.uniform(low=-1, high=1)
        b_i = rng.uniform(low=-np.pi, high=np.pi)
        # Numerical integration
        sol = scipy.integrate.solve_ivp(
            pendulum_ivp,
            t_span,
            y0=x0_i,
            method='RK45',
            t_eval=t,
            args=(a_i, b_i),
        )
        # Change of variables
        x = np.vstack([
            np.cos(sol.y[0, :]),
            np.sin(sol.y[0, :]),
            sol.y[1, :],
        ])
        # Form shifted and unshifted data matrices
        X_lst.append(x[:, :-1].T)
        Y_lst.append(x[:, 1:].T)
        u = sinusoidal_forcing(t, a_i, b_i)
        U_lst.append(u[np.newaxis, :-1].T)
    X = np.vstack(X_lst)
    Y = np.vstack(Y_lst)
    U = np.vstack(U_lst)

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(X[:, 0])
    ax[0].plot(X[:, 1])
    ax[0].plot(X[:, 2])
    ax[1].plot(Y[:, 0])
    ax[1].plot(Y[:, 1])
    ax[1].plot(Y[:, 2])
    ax[2].plot(U)
    plt.show()


if __name__ == '__main__':
    main()
