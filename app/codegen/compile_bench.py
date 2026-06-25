"""Compile-and-time benchmark for a unit-normalization kernel (week 4 optimization).

In the spirit of the course's "port Python to C++ and time it" exercise:
  1. Define a Python reference for unit-price normalization.
  2. Ask Groq to generate an equivalent C++ implementation (with a vetted fallback).
  3. If a C++ toolchain exists, compile and run it, check output equivalence vs Python,
     and compare latency over many iterations.
  4. If no compiler is available, return a documented runbook path instead of failing.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from app.llm import groq_client, router

# Test inputs: (price_inr, pack_size, unit_code) where unit_code 0=kg/litre/piece, 1=ml/g(per-100)
TEST_INPUTS = [(255.0, 5.0, 0), (142.0, 1.0, 0), (89.0, 150.0, 1), (67999.0, 1.0, 0),
               (449.0, 2.0, 0), (339.0, 400.0, 1)]


def py_unit_price(price: float, pack_size: float, unit_code: int) -> float:
    """Python reference: per-unit price, or per-100 for ml/g."""
    if pack_size <= 0:
        return 0.0
    if unit_code == 1:
        return round(price / pack_size * 100.0, 4)
    return round(price / pack_size, 4)


_CPP_FALLBACK = r"""
#include <cstdio>
#include <cstdlib>
#include <cmath>
double unit_price(double price, double pack, int code){
    if(pack <= 0.0) return 0.0;
    double r = (code==1) ? (price/pack*100.0) : (price/pack);
    return std::round(r*10000.0)/10000.0;
}
int main(int argc, char** argv){
    if(argc < 4){ return 1; }
    double price = atof(argv[1]);
    double pack  = atof(argv[2]);
    int    code  = atoi(argv[3]);
    printf("%.4f\n", unit_price(price, pack, code));
    return 0;
}
""".strip()


def generate_cpp(model: str | None = None) -> str:
    """Ask Groq to port the Python reference to C++ (vetted fallback on failure)."""
    spec = (
        "Write a complete C++ program (C++17) that reads three command-line args: "
        "price (double), pack_size (double), unit_code (int 0 or 1). Print to stdout the "
        "unit price with printf(\"%.4f\\n\", ...): if pack_size<=0 print 0.0000; if "
        "unit_code==1 compute price/pack_size*100 (per-100 for ml/g); else price/pack_size. "
        "Round to 4 decimals. Output ONLY C++ code, no fences."
    )
    try:
        code = groq_client.chat(
            [{"role": "system", "content": "You are a careful systems programmer. Output only C++ code."},
             {"role": "user", "content": spec}],
            model=model or router.route("codegen"), temperature=0.1, max_tokens=600)
        code = code.replace("```cpp", "").replace("```c++", "").replace("```", "").strip()
        if "int main" in code and "printf" in code:
            return code
    except Exception:  # noqa: BLE001
        pass
    return _CPP_FALLBACK


def _find_compiler() -> str | None:
    for cc in ("clang++", "g++", "c++"):
        if shutil.which(cc):
            return cc
    return None


RUNBOOK = (
    "No C++ toolchain found. To run the compiled benchmark:\n"
    "  macOS: xcode-select --install   (provides clang++)\n"
    "  Debian/Ubuntu: sudo apt-get install -y g++\n"
    "Then re-run: python -m app.codegen.compile_bench")


def benchmark(iterations: int = 200000, model: str | None = None) -> dict:
    cc = _find_compiler()
    cpp_src = generate_cpp(model)
    result = {"compiler": cc, "generated_chars": len(cpp_src),
              "used_fallback": cpp_src == _CPP_FALLBACK}
    if not cc:
        result.update({"status": "skipped_no_compiler", "runbook": RUNBOOK})
        return result

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "unit.cpp"
        binp = Path(td) / "unit"
        src.write_text(cpp_src)
        compile_t0 = time.time()
        try:
            subprocess.run([cc, "-O2", "-std=c++17", str(src), "-o", str(binp)],
                           check=True, capture_output=True, timeout=60)
        except Exception as e:  # noqa: BLE001 — generated code may not compile
            # retry with vetted fallback so the benchmark still produces a result
            src.write_text(_CPP_FALLBACK)
            result["used_fallback"] = True
            result["generated_compile_error"] = str(getattr(e, "stderr", e))[:300]
            subprocess.run([cc, "-O2", "-std=c++17", str(src), "-o", str(binp)],
                           check=True, capture_output=True, timeout=60)
        result["compile_s"] = round(time.time() - compile_t0, 3)

        # equivalence check
        mismatches = []
        for price, pack, code in TEST_INPUTS:
            py = py_unit_price(price, pack, code)
            try:
                out = subprocess.run([str(binp), str(price), str(pack), str(code)],
                                     capture_output=True, text=True, timeout=10).stdout.strip()
                cpp_val = float(out) if out else None
            except (subprocess.TimeoutExpired, ValueError):
                cpp_val = None
            if cpp_val is None or abs(cpp_val - py) > 1e-3:
                mismatches.append({"input": [price, pack, code], "py": py, "cpp": cpp_val})
        result["equivalent"] = not mismatches
        result["mismatches"] = mismatches

        # latency: Python loop vs compiled binary loop (binary times its own loop arg)
        price, pack, code = TEST_INPUTS[0]
        t0 = time.time()
        for _ in range(iterations):
            py_unit_price(price, pack, code)
        result["python_s"] = round(time.time() - t0, 4)
        # compiled: call once per 1000 to amortise process spawn, scaled estimate
        spawn_iters = max(1, iterations // 1000)
        t0 = time.time()
        for _ in range(spawn_iters):
            subprocess.run([str(binp), str(price), str(pack), str(code)],
                           capture_output=True, timeout=10)
        spawn_total = time.time() - t0
        result["cpp_proc_s_per_call"] = round(spawn_total / spawn_iters, 6)
        result["note"] = ("Python timed for an in-process loop; C++ timed per process "
                          "invocation (includes spawn overhead). Output equivalence is the "
                          "primary correctness signal.")
        result["status"] = "ok"
    return result


if __name__ == "__main__":
    print(json.dumps(benchmark(iterations=50000), indent=2))
