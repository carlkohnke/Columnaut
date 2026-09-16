"""Configurable multicomponent distillation simulation."""

from .configuration import ColumnConfig, load_config
from .components import Component, ComponentSet, load_component_database
from .results import SimulationBatch, SimulationResult
from .solver import ColumnSolver, simulate

__all__ = [
    "ColumnConfig",
    "ColumnSolver",
    "Component",
    "ComponentSet",
    "SimulationBatch",
    "SimulationResult",
    "load_config",
    "load_component_database",
    "simulate",
]

__version__ = "1.0.0"
