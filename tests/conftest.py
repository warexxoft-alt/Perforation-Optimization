"""tests/conftest.py — shared pytest fixtures."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


@pytest.fixture(scope="session")
def synthetic_df():
    """A modest synthetic + simulated dataset, shared read-only across tests
    that need real reservoir data (session-scoped so it's only computed once)."""
    from modules.data_gen import NigerDeltaDataGenerator
    from modules.simulator import CoupledSimulator
    df = NigerDeltaDataGenerator(200, seed=42).generate()
    df = CoupledSimulator().run_dataset(df)
    return df


@pytest.fixture
def rng():
    return np.random.default_rng(42)
