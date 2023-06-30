"""Design a data-driven controller for a pendulum via the Lie derivative.

This script accompanies the letter "Synthesizing Control Laws from Data using
Sum-of-Squares Optimization" by Jason J. Bramburger, Steven Dahdah, and
James R. Forbes.

The goal of this script is to use data-driven approximations of the Lie
derivative to identify optimal controllers from data. This script uses the
inverted pendulum on a cart model::

    theta'' = sin(theta) - eps * theta' - cos(theta) * u

where ``eps > 0`` is the friction coefficient and u is the control input. Here
the control is given by the acceleration of the cart.

This script identifies a state-dependent feedback controller that stabilizes
the pendulum in the upright position. The control law is identified through
polynomial optimization with a control Lyapunov function.

Application of the method transforms the state variables theta and theta'
to the 3D observables: ``x_1 = cos(theta)``, ``x_2 = sin(theta)``, and
``x_3 = theta'``. We then discover the controller as a polynomial function of
(x_1, x_2, x_3)``.
"""

import numpy as np
import picos
import scipy.integrate
import scipy.linalg
import SumOfSquares
import sympy
from matplotlib import pyplot as plt


def main():
    """Design a data-driven controller for a pendulum."""
    # Set random seed
    rng = np.random.default_rng(9234)

    # Max degree of x1 and x3 in phi dictionary of observables
    max_phi = 3
    # Max degree of x1 and x3 in psi dictionary of observables
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

    def pendulum_ivp(t, x, u):
        """Compute state derivative of pendulum."""
        x_dot = np.array([
            x[1],
            np.sin(x[0]) - 0.1 * x[1] - np.cos(x[0]) * u,
        ])
        return x_dot

    def pendulum_sine_ivp(t, x, a, b):
        """Compute state derivative of pendulum with sinusoidal forcing."""
        u = sinusoidal_forcing(t, a, b)
        x_dot = pendulum_ivp(t, x, u)
        return x_dot

    # Generate training data
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
            pendulum_sine_ivp,
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

    # Create Psi matrix
    pow_psi = []
    for p in SumOfSquares.basis_inhom(X.shape[1], max_psi):
        if p[1] < 2:
            pow_psi.append(p)
    Psi_no_u = np.vstack([np.prod(X**p, axis=1) for p in pow_psi]).T
    Psi = np.hstack([Psi_no_u, U * Psi_no_u])

    # Create Phi matrix
    pow_phi = []
    for p in SumOfSquares.basis_inhom(X.shape[1], max_phi):
        if p[1] < 2:
            pow_phi.append(p)
    Phi = np.vstack([np.prod(Y**p, axis=1) for p in pow_phi]).T
    # Solve for Koopman matrix
    K = scipy.linalg.lstsq(Psi, Phi)[0].T

    # Symbolic variables
    x1, x2, x3 = sympy.symbols('x_1, x_2, x_3')
    x = np.array([[x1], [x2], [x3]])
    w = np.vstack([np.prod(x.T**p, axis=1) for p in pow_psi])
    z = np.vstack([np.prod(x.T**p, axis=1) for p in pow_phi])

    # Controller basis
    ub = np.reshape(
        SumOfSquares.Basis.from_degree(
            X.shape[1],
            max_psi,
        ).to_sym([x1, x2, x3]),
        (-1, 1),
    )
    # Controller coefficients
    uc = np.reshape(
        sympy.symbols([f'u_{i}' for i in range(ub.shape[0])]),
        (-1, 1),
    )
    # Controller symbolic expression
    u = (uc.T @ ub).item()

    # Lyapunov function coefficients
    c = np.zeros_like(z)
    c[0, 0] = 1 + alpha  # 1
    c[3, 0] = -1  # x_1
    c[4, 0] = 0.5  # x_3^2
    c[15, 0] = -alpha  # x_1^3
    lyap = (c.T @ z).item()

    # Lie derivative approximation
    L = (K - np.eye(*K.shape)) / dt
    thresh = 0.05  # Use to stamp out noise
    L[np.abs(L) <= thresh] = 0

    # S-procedure polynomials
    s1 = SumOfSquares.poly_variable('s1', [x1, x2, x3], deg_u)
    s2 = SumOfSquares.poly_variable('s2', [x1, x2, x3], deg_u)
    # Main constraint
    q = w.shape[0]
    constr = (
        -c.T @ (L[:, :q] @ w + L[:, q:] @ w * u)  # Lie derivative
        + (1 - x[0, 0]**2 - x[1, 0]**2) * s1  # Trigononmetric constraint
        - (eta**2 - x[1, 0]**2) * s2  # Domain
    ).item()

    # SOS optimization problem
    prob = SumOfSquares.SOSProblem()
    prob.add_sos_constraint(constr, [x1, x2, x3])
    prob.add_sos_constraint(s2, [x1, x2, x3])
    # Controller coefficients as PICOS variables (instead of symbolic)
    ucv = picos.block([prob.sym_to_var(uc[i, 0]) for i in range(uc.shape[0])])
    prob.set_objective('min', picos.Norm(ucv, p=1))
    prob.solve(solver='mosek')

    # Get solution
    ucv = np.array(ucv)
    u = SumOfSquares.round_sympy_expr((ucv.T @ ub).item(), precision=4)
    print(f'Identified control input: u(x) = {u}')

    # Convert controller and Lyapunov functions to numpy
    u_numpy = sympy.lambdify([np.ravel(x)], u, 'numpy')
    lyap_numpy = sympy.lambdify([np.ravel(x)], lyap, 'numpy')

    def pendulum_controlled_ivp(t, x):
        """Compute state derivative of pendulum with SOS controller."""
        u = u_numpy(np.array([np.cos(x[0]), np.sin(x[0]), x[1]]))
        x_dot = pendulum_ivp(t, x, u)
        return x_dot

    # Numerically integrate trajectory with controller
    sol = scipy.integrate.solve_ivp(
        pendulum_controlled_ivp,
        t_span,
        y0=np.array([3, 0]),
        method='RK45',
        t_eval=t,
    )
    # Substitute in cosine and sine so we can evaluate V an u on the trajectory
    x_cos_sin = np.vstack([
        np.cos(sol.y[0, :]),
        np.sin(sol.y[0, :]),
        sol.y[1, :],
    ])

    # Plot pendulum trajectory
    fig, ax = plt.subplots()
    ax.plot(t, sol.y[0, :], label=r'$\theta(t)$')
    ax.plot(t, sol.y[1, :], '--', label=r'$\dot{\theta}(t)$')
    ax.set_xlim(0, 8)
    ax.set_ylim(-3, 3)
    ax.grid(ls='--')
    ax.set_xlabel('t')
    ax.legend(loc='upper right')

    # Plot Lyapunov function and input
    fig, ax = plt.subplots()
    ax.plot(
        t,
        lyap_numpy(x_cos_sin) / 10,
        label=r'$0.1 \cdot V(\theta(t), \dot{\theta}(t))$',
    )
    ax.plot(t, u_numpy(x_cos_sin), '--', label=r'$u(t)$')
    ax.set_xlim(0, 8)
    ax.set_ylim(-10, 20)
    ax.grid(ls='--')
    ax.set_xlabel('t')
    ax.legend(loc='upper right')

    plt.show()


if __name__ == '__main__':
    main()
