"""Small shared validation primitives for the public statistical API."""
from __future__ import annotations

import math


def number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def probability(value, name: str) -> float:
    result = number(value, name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be in (0,1)")
    return result


def count(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def series(values, name: str) -> list[float]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple of finite numbers")
    return [number(value, f"{name}[{index}]") for index, value in enumerate(values)]
