# Project-levitron
# Levitron Numerical Simulation

This repository contains a numerical simulation of a magnetically levitating spinning top (Levitron). The project combines analytical stability analysis with numerical integration of the equations of motion and numerical evaluation of the magnetic field. Detailed description of the mathematical model and numerical methods is provided in the accompanying bachelor thesis. Bachelor thesis can be found on XXX. If is script main.py runned with deafult parameters image same as reference_result.pdf should be plotted.

# Requirements

- Python 3.x
- NumPy
- Matplotlib
- Numba

Install required packages using:

```bash
pip install numpy matplotlib numba
```

# Repository structure

project/
├── main.py
├── computational.py/
│   ├── rk4
│   ├── get_B
│   ├── ring   
│   └── grad
└── stability_check.py/
    ├── stability_check
    ├── get_mass
    ├── upper_spin  
    └── lower_spin


