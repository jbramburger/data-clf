"""Design a data-driven controller for a pendulum via the Lie derivative."""

import sympy
import SumOfSquares


def main():
    """Design a data-driven controller for a pendulum."""
    # Quick test before startin
    x, y, z = sympy.symbols('x, y, z')
    M = (x**4 * y**2) + (x**2 * y**4) + z**6 - (3 * x**2 * y**2 * z**2)
    Mm = (x**2 + y**2 + z**2) * M
    prob = SumOfSquares.SOSProblem()
    c = prob.add_sos_constraint(Mm, [x, y, z])
    prob.solve()


if __name__ == '__main__':
    main()
