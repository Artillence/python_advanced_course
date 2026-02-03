# RUN FROM .py INSTEAD OF NOTEBOOK

from multiprocessing import Pool, cpu_count
import math
import random
import time

def compute_rms(values):
    """Compute RMS roughness of a list of heights."""
    s = 0.0
    for x in values:
        s += x * x
    return math.sqrt(s / len(values))

def main():
    # Create fake measurement data: 4 samples with 100_000 points each
    samples = [[random.uniform(-1e-6, 1e-6) for _ in range(100_000)] for _ in range(4)]

    print(f"Using up to {cpu_count()} CPU cores")
    start = time.perf_counter()
    with Pool() as pool:
        rms_values = pool.map(compute_rms, samples)
    elapsed = time.perf_counter() - start
    print(f"Total elapsed (threaded): {elapsed:.3f} s")
    
    print("RMS roughness per sample:")
    for i, rms in enumerate(rms_values, start=1):
        print(f"  Sample {i}: {rms:.3e} m")

if __name__ == "__main__":
    main()
