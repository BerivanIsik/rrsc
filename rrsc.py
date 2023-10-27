import math
import numpy as np
import time
from scipy.stats import ortho_group


def rand_orthog_cols(d, M):
    """
    Generate M columns of a random d x d orthogonal matrix
    """
    A = np.random.randn(d, M)
    Q, _ = np.linalg.qr(A)
    return Q


def lossy_DP_rrsc(X, eps, M, k, itr):
    (d, n) = X.shape
    exp_eps = np.exp(eps)
    X_perturb = np.zeros_like(X)
    # Generate the codebook.
    codebook_base = np.zeros((d, M))
    codebook_base[: M, :] = - 1 / np.sqrt(M * (M - 1))
    for code_idx in range(M):
        codebook_base[code_idx, code_idx] = (M - 1) / np.sqrt(M * (M - 1))

    for i in range(n):
        start_time = time.time()
        A = rand_orthog_cols(d, M)
        codebook = np.matmul(A, codebook_base[ : M, :])

        inner_product = np.matmul(X[:, i], codebook)
        sorted = np.argsort(inner_product)
        largest_k = sorted[ -k: ]
        probs = np.ones((M,))
        probs[largest_k] = exp_eps
        probs = probs / sum(probs)

        m_sampled = np.random.choice(a=np.arange(M), p=probs)
        if i == 0 and itr == 0:
            print(f"--- {time.time() - start_time:.5f} seconds for each client in RRSC ---")
        X_perturb[:, i] = codebook[:, m_sampled] #* sigma
    return X_perturb