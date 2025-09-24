from src.generate_data import generate_data, generate_mul_data, generate_gen_data
from src.subcase import TestCase, MulTestCase, GenTestCase
import numpy as np
import os

if not os.path.exists('data'):
    os.mkdir('data')

# configuration of data
samples_list = [500,1000,2000,5000,10000,15000,20000,25000]

latent, observed = 1, 3
for distribution in ['laplace', 'beta']:
    results = []
    for graph in ['a', 'b', 'c', 'd', 'e', 'f']:
        print(distribution, 'Fig. 3(', graph, ')')
        results.append([])
        for n_samples in samples_list:
            correct = 0
            for seed in range(100):
                data, weights, w_id = generate_data(graph, n_samples=n_samples, distribution=distribution, latent=latent, observed=observed, seed=seed)
                Z, T, O = data[:, 0], data[:, 1], data[:, 2]
                result = TestCase(T, O, Z)
                if graph in result:
                    correct += 1
            results[-1].append(1 - correct / 100)
        print(['sample size:erro',] + [f'{i}:{j:.2f}' for (i, j) in zip(samples_list, results[-1])]) # (sample size, error)
    np.save(f'data/{distribution}_discrimination.npy', results)

latent, observed = 2, 3
for distribution in ['laplace', 'beta']:
    results = []
    for graph in ['a', 'b', 'c', 'd', 'f', 'g', 'h']:
        print(distribution, 'Fig. 9(', graph, ')')
        results.append([])
        for n_samples in samples_list:
            correct = 0
            for seed in range(100):
                data, weights, w_id = generate_mul_data(graph, n_samples=n_samples, distribution=distribution, latent=latent, observed=observed, seed=seed)
                Z, T, O = data[:, 0], data[:, 1], data[:, 2]
                result = MulTestCase(T, O, Z)
                if graph in result:
                    correct += 1
            results[-1].append(1 - correct / 100)
        print(['sample size:erro',] + [f'{i}:{j:.2f}' for (i, j) in zip(samples_list, results[-1])]) # (sample size, error)
    np.save(f'data/mul_{distribution}_discrimination.npy', results)

latent, observed = 1, 3
for distribution in ['laplace', 'beta']:
    results = []
    for graph in ['d', 'f', 'h', 'i', 'm', 'n', 'q', 'r']:
        print(distribution, 'Fig. 5(', graph, ')')
        results.append([])
        for n_samples in samples_list:
            correct = 0
            for seed in range(100):
                data, weights, w_id = generate_gen_data(graph, n_samples=n_samples, distribution=distribution, latent=latent, observed=observed, seed=seed)
                Z, T, O = data[:, 0], data[:, 1], data[:, 2]
                result = GenTestCase(T, O, Z)
                if graph in result:
                    correct += 1
            results[-1].append(1 - correct / 100)
        print(['sample size:erro',] + [f'{i}:{j:.2f}' for (i, j) in zip(samples_list, results[-1])]) # (sample size, error)
    np.save(f'data/gen_{distribution}_discrimination.npy', results)