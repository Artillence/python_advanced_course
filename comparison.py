# RUN FROM A .py FILE (NOT DIRECTLY IN JUPYTER) FOR RELIABLE MULTIPROCESSING
#
# Example:
#   python compare_parallel_backends.py
#
# This script compares:
# - threading vs sequential for CPU-bound and I/O-bound tasks
# - multiprocessing.Pool for CPU-bound work
# - asyncio: correct (await asyncio.sleep) vs incorrect (time.sleep)
# - NumPy vs pure Python, and NumPy in threads (GIL released)
# - Numba vs pure Python, and Numba in threads (GIL released)
# - Optional CuPy vs NumPy vs pure Python (if CuPy is installed)

import time
import math
import sys
import threading
import asyncio
from multiprocessing import Pool, cpu_count
from typing import List

import numpy as np

# Optional imports
try:
    from numba import njit
except ImportError:
    njit = None

try:
    import cupy as cp
except ImportError:
    cp = None


# -------------------------------
# Utility helpers
# -------------------------------

def print_section(title: str) -> None:
    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_result(label: str, seconds: float) -> None:
    print(f"{label:<45} {seconds:8.3f} s")


# -------------------------------
# 1. CPU-bound and I/O-bound basic functions
# -------------------------------

def cpu_bound_python(n: int) -> float:
    """Simple CPU-bound loop: sum of squares."""
    s = 0.0
    for i in range(n):
        s += i * i
    return math.sqrt(s)


def io_bound_sleep(delay: float) -> None:
    """Simple I/O-bound stand-in: sleep to simulate waiting."""
    time.sleep(delay)


# For multiprocessing we need top-level worker functions

def cpu_worker(n: int) -> float:
    """Worker function for multiprocessing."""
    return cpu_bound_python(n)


# -------------------------------
# 2. Threading: CPU-bound vs I/O-bound
# -------------------------------

def benchmark_threading() -> None:
    print_section("1) Threading: CPU-bound vs I/O-bound")

    num_tasks_cpu = min(4, max(2, cpu_count()))
    num_tasks_io = num_tasks_cpu

    n_per_task = 300_000  # Adjust if this is too slow/fast
    delays = [0.4, 0.3, 0.6, 0.2][:num_tasks_io]

    # CPU-bound: sequential
    print("CPU-bound Python loop (sum of squares)")
    start = time.perf_counter()
    for _ in range(num_tasks_cpu):
        cpu_bound_python(n_per_task)
    t_seq = time.perf_counter() - start
    print_result("Sequential (CPU-bound, pure Python)", t_seq)

    # CPU-bound: threaded
    threads: List[threading.Thread] = []
    start = time.perf_counter()
    for _ in range(num_tasks_cpu):
        t = threading.Thread(target=cpu_bound_python, args=(n_per_task,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    t_thr = time.perf_counter() - start
    print_result("Threaded (CPU-bound, pure Python)", t_thr)

    if t_thr > 0:
        print(f"Speedup (threaded / sequential)          {t_seq / t_thr:8.3f} x")
    print("Expected: little or no speedup because of the GIL for CPU-bound Python.\n")

    # I/O-bound: sequential
    print("I/O-bound simulation (sleep)")
    start = time.perf_counter()
    for d in delays:
        io_bound_sleep(d)
    t_io_seq = time.perf_counter() - start
    print_result("Sequential (I/O-bound)", t_io_seq)

    # I/O-bound: threaded
    threads = []
    start = time.perf_counter()
    for d in delays:
        t = threading.Thread(target=io_bound_sleep, args=(d,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    t_io_thr = time.perf_counter() - start
    print_result("Threaded (I/O-bound)", t_io_thr)
    if t_io_thr > 0:
        print(f"Speedup (threaded / sequential)          {t_io_seq / t_io_thr:8.3f} x")
    print("Expected: near speedup by approx sum(delays) / max(delays).")


# -------------------------------
# 3. Multiprocessing for CPU-bound work
# -------------------------------

def benchmark_multiprocessing() -> None:
    print_section("2) Multiprocessing.Pool for CPU-bound work")

    num_tasks = min(4, max(2, cpu_count()))
    n_per_task = 5_000_000

    print(f"Using up to {num_tasks} processes out of {cpu_count()} CPU cores.")

    # Sequential
    start = time.perf_counter()
    for _ in range(num_tasks):
        cpu_worker(n_per_task)
    t_seq = time.perf_counter() - start
    print_result("Sequential (CPU-bound, pure Python)", t_seq)

    # Multiprocessing
    start = time.perf_counter()
    with Pool(processes=num_tasks) as pool:
        pool.map(cpu_worker, [n_per_task] * num_tasks)
    t_mp = time.perf_counter() - start
    print_result("Multiprocessing Pool (CPU-bound)", t_mp)

    if t_mp > 0:
        print(f"Speedup (multiprocessing / sequential)   {t_seq / t_mp:8.3f} x")
    print("Expected: clear speedup if per-task work is large enough.")


# -------------------------------
# 4. asyncio: correct vs incorrect sleeping
# -------------------------------

async def async_measure_blocking(delay: float) -> float:
    """Bad async: uses time.sleep, blocks event loop."""
    time.sleep(delay)  # This blocks the whole event loop
    return delay


async def async_measure_proper(delay: float) -> float:
    """Good async: uses asyncio.sleep, yields control."""
    await asyncio.sleep(delay)
    return delay


async def _async_benchmark_inner() -> None:
    delays = [0.1, 0.5, 0.2, 0.8, 0.3]

    print_section("3) asyncio: blocking vs non-blocking usage")

    # Bad case: async functions using time.sleep
    start = time.perf_counter()
    tasks = [asyncio.create_task(async_measure_blocking(d)) for d in delays]
    await asyncio.gather(*tasks)
    t_blocking = time.perf_counter() - start
    print_result("asyncio + time.sleep (anti-pattern)", t_blocking)

    # Good case: async functions using asyncio.sleep
    start = time.perf_counter()
    tasks = [asyncio.create_task(async_measure_proper(d)) for d in delays]
    await asyncio.gather(*tasks)
    t_proper = time.perf_counter() - start
    print_result("asyncio + await asyncio.sleep()", t_proper)

    if t_proper > 0:
        print(f"Speedup (proper / blocking)             {t_blocking / t_proper:8.3f} x")
    print("Expected: proper async time ~ max(delay), blocking async time ~ sum(delays).")


def benchmark_asyncio() -> None:
    # Run the async benchmark in a fresh event loop
    asyncio.run(_async_benchmark_inner())


# -------------------------------
# 5. NumPy vs pure Python, and NumPy in threads
# -------------------------------

def cpu_numpy(arr: np.ndarray) -> float:
    # Simple computation that forces a pass over the data
    return float(np.sqrt(np.sum(arr * arr)))


def cpu_python_from_array(arr: np.ndarray) -> float:
    s = 0.0
    # iterate over a Python list copy to emphasize Python loop overhead
    for x in arr.tolist():
        s += x * x
    return math.sqrt(s)


def benchmark_numpy() -> None:
    print_section("4) NumPy vs pure Python, and NumPy in threads")

    n = 10_000_000
    num_tasks = min(4, max(2, cpu_count()))
    print(f"Using array size n = {n}, tasks = {num_tasks}")

    base_arr = np.random.normal(size=n)

    # Pure Python
    start = time.perf_counter()
    cpu_python_from_array(base_arr)
    t_py = time.perf_counter() - start
    print_result("Pure Python loop (single task)", t_py)

    # NumPy single task
    start = time.perf_counter()
    cpu_numpy(base_arr)
    t_np = time.perf_counter() - start
    print_result("NumPy vectorized (single task)", t_np)

    # NumPy sequential multi-task
    arrays = [np.random.normal(size=n) for _ in range(num_tasks)]
    start = time.perf_counter()
    for a in arrays:
        cpu_numpy(a)
    t_np_seq = time.perf_counter() - start
    print_result("NumPy multi-task (sequential)", t_np_seq)

    # NumPy threaded multi-task
    threads: List[threading.Thread] = []
    start = time.perf_counter()
    for a in arrays:
        t = threading.Thread(target=cpu_numpy, args=(a,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    t_np_thr = time.perf_counter() - start
    print_result("NumPy multi-task (threaded)", t_np_thr)

    if t_np_thr > 0:
        print(f"Speedup (threaded / sequential)         {t_np_seq / t_np_thr:8.3f} x")
    print("Expected: NumPy should be much faster than pure Python,")
    print("and multiple independent NumPy calls can scale with threads because NumPy releases the GIL.")


# -------------------------------
# 6. Numba vs pure Python, and Numba in threads
# -------------------------------

if njit is not None:
    def cpu_python(n) -> float:
        s = 0.0
        for i in range(n):
            s += i * i
        return math.sqrt(s)
    
    @njit(nogil=True)
    def cpu_numba(n: int) -> float:
        s = 0.0
        for i in range(n):
            s += i * i
        return math.sqrt(s)
else:
    cpu_numba = None  # type: ignore


def benchmark_numba() -> None:
    print_section("5) Numba vs pure Python, and Numba in threads")

    if cpu_numba is None:
        print("Numba is not installed. Skipping Numba benchmarks.")
        return

    n = 10_000_000
    num_tasks = min(4, max(2, cpu_count()))

    # Pure Python
    start = time.perf_counter()
    cpu_python(n)
    t_py = time.perf_counter() - start
    print_result("Pure Python loop (single task)", t_py)

    # Numba single task: first call (includes JIT compile)
    start = time.perf_counter()
    cpu_numba(n)
    t_nb_first = time.perf_counter() - start
    print_result("Numba single task (first call - compile)", t_nb_first)

    # Numba single task: second call (fast)
    start = time.perf_counter()
    cpu_numba(n)
    t_nb = time.perf_counter() - start
    print_result("Numba single task (warm)", t_nb)

    arrays = [n for _ in range(num_tasks)]

    # Numba sequential multi-task
    start = time.perf_counter()
    for a in arrays:
        cpu_numba(a)
    t_nb_seq = time.perf_counter() - start
    print_result("Numba multi-task (sequential)", t_nb_seq)

    # Numba threaded multi-task
    threads: List[threading.Thread] = []
    start = time.perf_counter()
    for a in arrays:
        t = threading.Thread(target=cpu_numba, args=(a,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    t_nb_thr = time.perf_counter() - start
    print_result("Numba multi-task (threaded)", t_nb_thr)

    if t_nb > 0:
        print(f"Speedup (Numba / pure Python, single)   {t_py / t_nb:8.3f} x")
    if t_nb_thr > 0:
        print(f"Speedup (threaded / sequential, Numba)  {t_nb_seq / t_nb_thr:8.3f} x")
    print("Expected: Numba should drastically beat pure Python for large arrays,")
    print("and multiple Numba kernels can run in parallel threads because Numba releases the GIL.")


# -------------------------------
# 7. Optional CuPy vs NumPy: heavy pairwise distances
# -------------------------------

def benchmark_cupy() -> None:
    print_section("6) CuPy vs NumPy vs pure Python (optional GPU)")

    if cp is None:
        print("CuPy is not installed or no GPU is available. Skipping CuPy benchmark.")
        return

    # Size and iterations chosen so that:
    # - NumPy is clearly faster than pure Python
    # - CuPy has enough math work to amortize GPU overhead and usually win
    n = 20_000_000
    iters = 10
    print(f"Array size for this benchmark: n = {n}, iterations = {iters}")

    # -------------------------------
    # Pure Python baseline (simple work)
    # -------------------------------
    arr_py = np.random.normal(size=n)

    start = time.perf_counter()
    cpu_python_from_array(arr_py)
    t_py = time.perf_counter() - start
    print_result("Pure Python loop (CPU)", t_py)

    # -------------------------------
    # Heavy NumPy compute
    # -------------------------------
    arr_np = np.random.normal(size=n).astype(np.float32)

    def heavy_numpy(x: np.ndarray, iters: int) -> float:
        # Repeat a non-trivial math kernel several times
        for _ in range(iters):
            x = np.sin(x) + np.cos(x) * 1.000001 + np.sqrt(x * x + 0.1234)
        return float(x.mean())

    start = time.perf_counter()
    mean_np = heavy_numpy(arr_np, iters)
    t_np = time.perf_counter() - start
    print_result("NumPy (CPU, heavy trig loop)", t_np)

    # -------------------------------
    # Heavy CuPy compute (GPU)
    # -------------------------------
    # Transfer once, then ONLY measure math-heavy GPU work.
    arr_gpu = cp.asarray(arr_np)

    def heavy_cupy(x: "cp.ndarray", iters: int) -> float:
        for _ in range(iters):
            x = cp.sin(x) + cp.cos(x) * 1.000001 + cp.sqrt(x * x + 0.1234)
        return float(x.mean())  # brings back a single scalar

    # Just to see the copy cost, not included in t_gpu:
    start_copy = time.perf_counter()
    _ = cp.asarray(arr_np)
    t_copy = time.perf_counter() - start_copy
    print_result("Host -> GPU copy (one large array)", t_copy)

    # Now time only the compute-heavy kernel
    start = time.perf_counter()
    mean_gpu = heavy_cupy(arr_gpu, iters)
    t_gpu = time.perf_counter() - start
    print_result("CuPy (GPU, heavy trig loop)", t_gpu)

    # -------------------------------
    # Speedups
    # -------------------------------
    if t_np > 0:
        print(f"Speedup (NumPy / Python) {t_py / t_np:8.3f} x")
    if t_gpu > 0:
        print(f"Speedup (CuPy compute / NumPy compute)    {t_np / t_gpu:8.3f} x")

    print("Note: Here the GPU does repeated trig and sqrt operations,")
    print("so the work is compute-heavy rather than copy-heavy.")
    print("On a decent GPU, CuPy should clearly beat NumPy on the heavy loop above.")

# -------------------------------
# Main entry point
# -------------------------------

def main() -> None:
    print("Python executable:", sys.executable)
    print("Python version:   ", sys.version.split()[0])
    print("Detected CPU cores:", cpu_count())
    print("Numba available:", bool(njit))
    print("CuPy available:", cp is not None)
    print()

    benchmark_threading()
    benchmark_multiprocessing()
    benchmark_asyncio()
    benchmark_numpy()
    benchmark_numba()
    benchmark_cupy()

    print("\nAll benchmarks finished.")


if __name__ == "__main__":
    main()
