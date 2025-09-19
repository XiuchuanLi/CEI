import numpy as np
import os
from src.methods import cross_moment
from src.generate_data import generate_data
from src.utils import performance

if not os.path.exists('data'):
    os.mkdir('data')

# hyper-parameters of cross-moments
deg = 3
samples_list = [500,1000,2000,5000,10000,15000,20000,25000]


latent, observed = 1, 3 
for distribution in ['laplace', 'uniform']:
    results = [[], []] # mean, std
    print(distribution, 'Fig. 3(a)')
    for n_samples in samples_list:
        weight_CM_pred, weight_CM_true = [], []
        for seed in range(100):
            data, g, weights, w_id = generate_data('a', n_samples=n_samples, distribution=distribution, latent=latent, observed=observed, seed=seed)
            
            Z = data[:, 0].numpy()
            D = data[:, 1].numpy()
            Y = data[:, 2].numpy()
            Z -= Z.mean()
            D -= D.mean()
            Y -= Y.mean()
            tmp_pred = cross_moment(Z, D, Y, deg)
            weight_CM_pred.append(tmp_pred)
            weight_CM_true.append(weights[w_id].item())
        
        weight_CM_true = np.array(weight_CM_true)
        weight_CM_pred = np.array(weight_CM_pred)
        error_CM = np.abs((weight_CM_true - weight_CM_pred) / weight_CM_true)
        mean, std = performance(error_CM)
        results[0].append(mean)
        results[1].append(std)
        print(f'{n_samples}: {mean:.2f}±{std:.2f}')
    np.save(f'data/cm_{distribution}_a.npy', results)
