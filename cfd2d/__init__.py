"""cfd2d: 2D-CFD-Löser für inkompressible Strömungen in Python.

Finite-Volumen-Diskretisierung auf versetztem MAC-Gitter, Projektionsmethode
nach Chorin, Numba-beschleunigte Kernel, Druckgleichung über SciPy (dünn
besetzt, LU oder CG), Live-Visualisierung mit Matplotlib.

Schnellstart::

    from cfd2d import cases
    case = cases.lid_driven_cavity(re=100, n=64)
    case.sim.run(case.t_end, steady_tol=case.steady_tol)

Interaktiv::

    python -m cfd2d
"""

from .boundary import BoundaryCondition
from .grid import Grid
from .solver import Simulation, StopSimulation

__version__ = "1.0.0"
__all__ = ["Grid", "BoundaryCondition", "Simulation", "StopSimulation", "__version__"]
