# RUN FROM .py INSTEAD OF NOTEBOOK

from multiprocessing import Pool
import random

# TODO: compute average diameter per batch in parallel.

def average(values):
    # return ...

# Create example batches of diameters in mm
batches = [
    [random.uniform(9.95, 10.05) for _ in range(50)],
    [random.uniform(4.95, 5.05) for _ in range(80)],
    [random.uniform(19.9, 20.1) for _ in range(100)],
]

def main():
    # with Pool() as pool:
    #     averages = ...
    # for i, avg in enumerate(averages, start=1):
    #     print(f"Batch {i}: average diameter = {avg:.3f} mm")

if __name__ == "__main__":
    main()
