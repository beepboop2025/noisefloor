"""Seeded real-symmetric matrix Brownian motion, for synthetic research only."""
from __future__ import annotations

import math
import random

from ._linalg import eigh
from .market import _hash
from .schemas import DYSON_REQUEST, validate


def simulate(*, dimension: int = 6, steps: int = 40, dt: float = 0.05, seed: int = 0) -> dict:
    """Sample the beta=1 eigenvalue process by evolving the underlying matrix.

    Off-diagonal increment variance is dt/N, diagonal variance 2*dt/N.
    Matrix increments avoid numerical integration of singular 1/(lambda_i-lambda_j)
    drifts. This is an unconfined Brownian reference, not a covariance forecast.
    """
    request = {"dimension": dimension, "steps": steps, "dt": dt, "seed": seed}
    validate(request, DYSON_REQUEST)
    rng = random.Random(seed)
    matrix = [[0.0] * dimension for _ in range(dimension)]
    off, diagonal = math.sqrt(dt / dimension), math.sqrt(2 * dt / dimension)
    path = [{"time": 0.0, "eigenvalues": [0.0] * dimension}]
    for step in range(1, steps + 1):
        for i in range(dimension):
            matrix[i][i] += rng.gauss(0, diagonal)
            for j in range(i):
                matrix[i][j] += rng.gauss(0, off)
                matrix[j][i] = matrix[i][j]
        values, _vectors = eigh(matrix)
        path.append({"time": step * dt, "eigenvalues": values})
    return {"schema": "noisefloor.dyson-reference.v1", "status": "synthetic",
            "maturity": "research", "validity": "synthetic_reference", "guarantee": None,
            "parameters": request, "beta": 1, "path": path,
            "initial_condition": "zero_matrix", "mean_reversion": False,
            "normalization": "off_diagonal_variance=dt/N; diagonal_variance=2*dt/N",
            "market_calibrated": False, "is_covariance_model": False,
            "execution_authority": False, "request_sha256": _hash(request),
            "limits": ["Synthetic eigenvalue repulsion reference; not market observations, alpha or a risk forecast.",
                       "A real-symmetric Brownian matrix can have negative eigenvalues and is not a covariance matrix.",
                       "Ordered eigenvalues at discrete sample times are not tracked economic factor identities.",
                       "The seed enables replay within a recorded runtime; it does not establish empirical calibration."]}
