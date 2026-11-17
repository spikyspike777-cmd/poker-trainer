"""Солверы постфлопа за единым интерфейсом (см. base.py)."""
from .base import ActionFreq, Solver, SolverResult, Spot
from .river_cfr import RiverCFR
from .rust_cfr import RustCFR


def default_solver() -> Solver:
    """Солвер на Rust, если расширение собрано; иначе солвер ривера на Python.

    Если в параметрах включено value_net и обученная сеть лежит на месте, тёрн решается с ней.
    """
    if not RustCFR.available():
        return RiverCFR(iterations=300)
    solver = RustCFR()
    from ..params import P
    if P.get("value_net"):
        from .. import valuenet
        solver.value_net = valuenet.load()
    return solver


__all__ = ["ActionFreq", "Solver", "SolverResult", "Spot", "RiverCFR", "RustCFR", "default_solver"]
