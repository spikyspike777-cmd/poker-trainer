"""Солверы постфлопа за единым интерфейсом (см. base.py)."""
from .base import ActionFreq, Solver, SolverResult, Spot
from .river_cfr import RiverCFR
from .rust_cfr import RustCFR


def default_solver() -> Solver:
    """Солвер на Rust (тёрн и ривер), если расширение собрано; иначе солвер ривера на Python."""
    return RustCFR() if RustCFR.available() else RiverCFR(iterations=300)


__all__ = ["ActionFreq", "Solver", "SolverResult", "Spot", "RiverCFR", "RustCFR", "default_solver"]
