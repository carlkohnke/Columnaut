from __future__ import annotations

from pathlib import Path

import pytest

from distillation.configuration import load_config
from distillation.solver import simulate


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def hydrocarbon_config():
    return load_config(ROOT / "examples" / "hydrocarbon_column.yaml")


@pytest.fixture(scope="session")
def hydrocarbon_batch(hydrocarbon_config):
    return simulate(hydrocarbon_config)
