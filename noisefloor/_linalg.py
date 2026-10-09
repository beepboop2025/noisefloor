"""Bounded symmetric Jacobi eigensolver for the dependency-free research tools.

At most 32 dimensions. Eigenvectors are returned as columns, in descending
eigenvalue order. Nonconvergence raises instead of returning a partial spectrum.
"""
from __future__ import annotations

import math


def eigh(matrix):
    n = len(matrix)
    if not 1 <= n <= 32 or any(len(row) != n for row in matrix):
        raise ValueError("symmetric matrix must be square with dimension 1..32")
    if any(not math.isfinite(x) for row in matrix for x in row):
        raise ValueError("matrix must be finite")
    scale = max(abs(x) for row in matrix for x in row) or 1.0
    a = [[float(x) / scale for x in row] for row in matrix]
    if any(abs(a[i][j] - a[j][i]) > 1e-12 for i in range(n) for j in range(i)):
        raise ValueError("matrix must be symmetric")
    vectors = [[float(i == j) for j in range(n)] for i in range(n)]
    tolerance = 1e-14
    for _ in range(50):
        if max((abs(a[i][j]) for i in range(n) for j in range(i)), default=0) <= tolerance:
            order = sorted(range(n), key=lambda k: a[k][k], reverse=True)
            values = [a[k][k] * scale for k in order]
            if any(not math.isfinite(x) for x in values):
                raise ValueError("eigenvalue overflow; rescale the input")
            return values, [[row[k] for k in order] for row in vectors]
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) <= tolerance:
                    continue
                tau = (a[q][q] - a[p][p]) / (2 * apq)
                t = math.copysign(1.0, tau) / (abs(tau) + math.hypot(1.0, tau))
                c = 1.0 / math.sqrt(1 + t * t)
                s = t * c
                a[p][p] -= t * apq
                a[q][q] += t * apq
                a[p][q] = a[q][p] = 0.0
                for k in range(n):
                    if k != p and k != q:
                        kp, kq = a[k][p], a[k][q]
                        a[k][p] = a[p][k] = c * kp - s * kq
                        a[k][q] = a[q][k] = s * kp + c * kq
                    vp, vq = vectors[k][p], vectors[k][q]
                    vectors[k][p], vectors[k][q] = c * vp - s * vq, s * vp + c * vq
    raise ValueError("symmetric eigensolver did not converge")


def normalized(values):
    """Unit-norm centered values without squaring the original input scale."""
    magnitude = max(abs(x) for x in values)
    if not magnitude:
        return None
    scaled = [x / magnitude for x in values]
    mean = math.fsum(scaled) / len(scaled)
    deviations = [x - mean for x in scaled]
    norm = math.sqrt(math.fsum(x * x for x in deviations))
    return [x / norm for x in deviations] if norm else None


def gram(columns):
    return [[math.fsum(x * y for x, y in zip(a, b)) for b in columns] for a in columns]
