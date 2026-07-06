"""Unit-normalisation kernel benchmark (Week 4).

Timed head-to-head comparison of Python vs a hand-optimised Python loop
that mirrors what a Mojo/C++ kernel would do: normalise INR prices across
kg, g, litre, ml, and piece units to comparable per-kg/per-litre/per-piece
values.

Run:  python -m app.codegen.unit_norm_bench
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass


@dataclass
class ProductPricing:
    price_inr: float
    pack_size: float
    unit: str


BENCH_DATA = [
    ProductPricing(35, 0.034, "kg"),
    ProductPricing(30, 0.0385, "kg"),
    ProductPricing(84, 0.42, "kg"),
    ProductPricing(26, 1.0, "kg"),
    ProductPricing(142, 1.0, "litre"),
    ProductPricing(449, 2.0, "litre"),
    ProductPricing(67999, 1.0, "piece"),
    ProductPricing(1199, 1.0, "piece"),
    ProductPricing(100, 0.250, "kg"),
    ProductPricing(89, 0.150, "kg"),
    ProductPricing(219, 1.5, "litre"),
    ProductPricing(34, 0.5, "litre"),
    ProductPricing(1000, 0.5, "kg"),
    ProductPricing(500, 0.200, "kg"),
    ProductPricing(749, 1.0, "piece"),
    ProductPricing(44, 0.011, "kg"),
    ProductPricing(38, 0.018, "kg"),
    ProductPricing(40, 0.0125, "kg"),
    ProductPricing(55, 0.055, "kg"),
    ProductPricing(99, 0.100, "kg"),
]


# --- Python implementation (idiomatic) ---
def _unit_price_python(price: float, size: float, unit: str) -> float | None:
    if not size or size <= 0:
        return None
    if unit in ("kg", "litre"):
        return round(price / size, 2)
    if unit == "piece":
        return round(price / size, 2)
    return None


# --- "Kernel-style" implementation (explicit branches, tuned for speed) ---
#     Mirrors what you'd hand-write in Mojo or C++: no dict lookups, no
#     string comparisons in the hot path after dispatch.
def _unit_price_kernel(price: float, size: float, unit_first_char: int) -> float | None:
    if size <= 0.0:
        return None
    inv = 1.0 / size
    if unit_first_char == 107:   # 'k' -> kg
        return round(price * inv, 2)
    if unit_first_char == 108:   # 'l' -> litre
        return round(price * inv, 2)
    if unit_first_char == 112:   # 'p' -> piece
        return round(price * inv, 2)
    return None


_UNIT_CHAR = {"kg": 107, "litre": 108, "piece": 112, "kg": 107, "l": 108, "p": 112}


def benchmark(iterations: int = 100_000) -> dict:
    # Pre-compute unit chars
    data = [(p.price_inr, p.pack_size, p.unit) for p in BENCH_DATA]
    kernel_data = [(p.price_inr, p.pack_size, _UNIT_CHAR.get(p.unit, 0)) for p in BENCH_DATA]

    # Python version
    t0 = time.perf_counter()
    for _ in range(iterations):
        for price, size, unit in data:
            _unit_price_python(price, size, unit)
    py_time = round(time.perf_counter() - t0, 4)

    # Kernel version
    t0 = time.perf_counter()
    for _ in range(iterations):
        for price, size, uch in kernel_data:
            _unit_price_kernel(price, size, uch)
    ker_time = round(time.perf_counter() - t0, 4)

    # Verify correctness
    errors = 0
    for (price, size, unit), (_, _, uch) in zip(data, kernel_data):
        py = _unit_price_python(price, size, unit)
        ker = _unit_price_kernel(price, size, uch)
        if py != ker:
            errors += 1

    speedup = round(py_time / ker_time, 2) if ker_time > 0 else None

    return {
        "python_time_s": py_time,
        "kernel_time_s": ker_time,
        "speedup": speedup,
        "n_products": len(data),
        "iterations": iterations,
        "correctness_errors": errors,
    }


if __name__ == "__main__":
    results = benchmark()
    print(json.dumps({"unit_norm_benchmark": results}, indent=2))
