from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as ss

import pickle


with open('pickled/500-600geom(0.5)1e6immigration.pickle', 'rb') as f:
    results = pickle.load(f)


bounds = {'a': (0.1, 100), 'scale': (0.1, 100)}
fitted_params = ss.fit(ss.gamma, results, bounds=bounds).params
shape_estimated, scale_estimated = fitted_params.a, fitted_params.scale

print(f"Estimated shape (a): {shape_estimated:.2f}, scale: {scale_estimated:.2f}")

plt.figure(figsize=(10, 3))
plt.hist(results, bins=1000, density=True, color='green', alpha=0.6, label='Observed Data')

t = np.linspace(min(results), max(results), 1000)
plt.plot(t, ss.gamma.pdf(t, a=shape_estimated, scale=scale_estimated),
         color='blue', linewidth=1, label=f'Gamma Fit (a={shape_estimated:.2f}, scale={scale_estimated:.2f})')

plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Gamma Distribution Fit to Immigration Data')
plt.legend()
plt.show()
