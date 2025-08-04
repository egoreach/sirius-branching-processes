from scipy.stats import randint, binom, geom
from matplotlib import pyplot as plt

import concurrent.futures
from pickle import dump
from tqdm import tqdm


from sys import setrecursionlimit
setrecursionlimit(2 * 10**9)


v, u = 500, 600
N = 3 * 10 ** 5

it = randint(v, u)
X, Y = geom(0.5), geom(0.5)


def W(n):
    s = 1

    for i in range(n):
        s = X.rvs(s).sum() - s

        if s == 0:
            return W(n)

    return (s / n).item()


def main():
    tasks = it.rvs(N)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = []

        with tqdm(total=len(tasks), desc="Branching...") as pbar:
            futures = {executor.submit(W, task): task for task in tasks}

            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                pbar.update(1)

    return results


if __name__ == "__main__":
    fname = "500-600geom(0.5)3e5,WO immigration"
    results = main()

    with open(f'{fname}.pickle', 'wb') as f:
        dump(results, f)

    plt.figure(figsize=(10, 3))
    plt.hist(results, bins=50, density=True)
    plt.show()
