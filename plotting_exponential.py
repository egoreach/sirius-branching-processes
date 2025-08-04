from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as ss
import pickle

with open('pickled/500-600geom(0.5)3e5,WO immigration.pickle', 'rb') as f:
    results = pickle.load(f)

bounds = {'scale': (0.1, 100)}
fitted_params = ss.fit(ss.expon, results, bounds=bounds).params
scale_estimated = fitted_params.scale
lambda_estimated = 1/scale_estimated

print(f"Estimated lambda (rate): {lambda_estimated:.4f}, scale: {scale_estimated:.4f}")

plt.figure(figsize=(10, 3))
plt.hist(results, bins=1000, density=True, color='green', alpha=0.6, label='Observed Data')

t = np.linspace(min(results), max(results), 1000)
plt.plot(t, ss.expon.pdf(t, scale=scale_estimated),
         color='blue', linewidth=1, label=f'Exponential Fit (λ={lambda_estimated:.4f}, scale={scale_estimated:.4f})')

plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Exponential Distribution Fit to Immigration Data')
plt.legend()
plt.show()