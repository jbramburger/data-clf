# Synthesizing Control Laws from Data using Sum-of-Squares Optimization

This repository contains MATLAB and Python scripts to reproduce the data and
figures from [Synthesizing Control Laws from Data using Sum-of-Squares
Optimization](https://arxiv.org/abs/2303.01483) by [Jason J.
Bramburger](https://hybrid.concordia.ca/jbrambur/), [Steven
Dahdah](https://github.com/sdahdah), and [James R.
Forbes](https://www.decar.ca/) (2023).

## Paper Abstract

The control Lyapunov function (CLF) approach to nonlinear control design is
well established. Moreover, when the plant is control affine and polynomial,
sum-of-squares (SOS) optimization can be used to find a polynomial controller
as a solution to a semidefinite program. This letter considers the use of
data-driven methods to design a polynomial controller by leveraging Koopman
operator theory, CLFs, and SOS optimization. First, Extended Dynamic Mode
Decomposition (EDMD) is used to approximate the Lie derivative of a given CLF
candidate with polynomial lifting functions. Then, the polynomial Koopman model
of the Lie derivative is used to synthesize a polynomial controller via SOS
optimization. The result is a flexible data-driven method that skips the
intermediary process of system identification and can be applied widely to
control problems. The proposed approach is used to successfully synthesize a
controller to stabilize an inverted pendulum on a cart.

## Setup

### MATLAB

`pend_control.m` requires YALMIP and MOSEK to run. Both packages can be
download for free at:

- YALMIP: https://yalmip.github.io/download/
- MOSEK: https://www.mosek.com/downloads/

### Python

To use `pend_control.py`, install the requirements in a virtual environment
using:

```bash
(venv) $ pip install -r ./python/requirements.txt
```

### MOSEK

The MOSEK solver is required for both the MATLAB and Python scripts. Personal
academic licenses are available
[here](https://www.mosek.com/products/academic-licenses/). The license file
must be placed in `~/mosek/mosek.lic` on Linux/Mac or
`C:\Users\<USER>\mosek\mosek.lic` on Windows.
