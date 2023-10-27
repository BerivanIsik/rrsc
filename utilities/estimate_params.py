import numpy as np
from utilities.get_parameters import *
from utilities.optimize_unbias import *
import sys


def estimate_first_order_expectation(d, M):
    T = 1000000
    a = np.random.normal(0, 1, (T, d))
    a /= np.reshape(np.linalg.norm(a, axis=1), [-1, 1])
    a_first_M = a[:, : M]
    largest = np.amax(a_first_M, axis=1)
    estimate = np.sum(largest) / T
    return estimate

def estimate_k_and_bias(d, M, eps):
    exp_eps = np.exp(eps)
    T = 1000000

    codebook = np.zeros((d, M))
    codebook[: M, :] = - 1 / np.sqrt(M * (M - 1))
    for code_idx in range(M):
        codebook[code_idx, code_idx] = (M - 1) / np.sqrt(M * (M - 1))

    a = np.random.normal(0, 1, (T, d))
    a /= np.reshape(np.linalg.norm(a, axis=1), [-1, 1])

    inner_product = np.matmul(a[:, :M], codebook[:M, :])
    best_k = 0
    best_bias = sys.maxsize
    for k in range(1, 50):
        if k >= M:
            break
        sorted = np.sort(inner_product, axis=1)
        largest_k = sorted[ :, -k: ]
        estimate = np.sum(largest_k) / T
        bias = (k * exp_eps + (M - k)) / (
                exp_eps - 1) / estimate
        if bias < best_bias:
            best_k = k
            best_bias = bias
    print('Best k is {}.'.format(best_k))
    return best_k, best_bias

def generate_uniform_sphere_codebook(m, d, num_repeat):
    # generate m codewords of d-dimension (repeat num_repeat times)
    codebook = np.random.randn(m, d, num_repeat)
    row_norm = np.linalg.norm(codebook, axis=1, keepdims=True)
    # normalize codewords (now uniform on sphere)
    codebook = codebook / row_norm
    return codebook