from mmrc import *
import numpy as np
from privunitG import *
from rrsc import *
import os
from scipy.stats import ortho_group
import scipy.io as io
from sqkr import *
from utilities.estimate_params import *
from utilities.modify_pi import *


def rrsc_comparison(vary='bitrate', num_itr=10, d_list=[500], n=5000, eps_list=[6]):
    '''
    vary: specifies which parameter to sweep through. Options: bitrate, users, eps, d
    '''

    for d in d_list:
        if not vary in ["users", "d"]:
            # Generate input.
            input = np.zeros((num_itr, d, n))
            for itr in range(num_itr):
                # Generate data matrix
                X = np.zeros((d, n))
                for j in range(int(n / 2)):
                    v_1 = np.concatenate([np.random.normal(10, 1, int(d))])
                    v_2 = np.concatenate([np.random.normal(1, 1, int(d))])
                    X[:, j] = v_1 / np.linalg.norm(v_1)
                    X[:, j + int(n / 2)] = v_2 / np.linalg.norm(v_2)
                input[itr] = X
        for eps in eps_list:
            bitrate = eps
            if vary == 'users':
                step_num = 5
                indices = [2000, 4000, 6000, 8000, 10000]
            elif vary == 'eps':
                step_num = 8
                indices = [1, 2, 3, 4, 5, 6, 7, 8]
            elif vary == 'bitrate':
                step_num = 8
                indices = [1, 2, 3, 4, 5, 6, 7, 8]
            elif vary == 'd':
                step_num = 5
                indices = [200, 400, 600, 800, 1000]

            M_rrsc = int(2**bitrate) # number of codewords for our method.
            budget = get_optimized_budget(eps, d)
            coding_cost_mmrc = bitrate
            k_equiv = bitrate
            mse_dict = {"sqkr": np.zeros(step_num),
                        "mmrc": np.zeros(step_num),
                        "privunit": np.zeros(step_num),
                        "rrsc": np.zeros(step_num)
                        }
            run_mse = {"sqkr": np.zeros([step_num, num_itr]),
                        "mmrc": np.zeros([step_num, num_itr]),
                        "privunit": np.zeros([step_num, num_itr]),
                        "rrsc": np.zeros([step_num, num_itr])
                        }
            bitrate_dict = {"sqkr": eps,
                            "mmrc": coding_cost_mmrc,
                            "privunit": 64,
                            "rrsc": int(np.ceil(np.log2(M_rrsc)))
                        }

            print(
                f"d = {d}, eps = {eps}, rate_SQKR = {eps}, rate_MMRC = {coding_cost_mmrc}, rate_rrsc = {int(np.ceil(np.log2(M_rrsc)))}"
            )

            for step in range(step_num):
                if vary == 'users':
                    n = indices[step]
                    best_k, best_bias = estimate_k_and_bias(d, M_rrsc, eps)
                elif vary == 'd':
                    d = indices[step]
                    best_k, best_bias = estimate_k_and_bias(d, M_rrsc, eps)
                else:
                    if vary == 'eps':
                        eps = indices[step]
                        budget = get_optimized_budget(eps, d)
                        bitrate = eps
                        M_rrsc = int(2 ** bitrate)  # number of codewords for LDP-RD
                        coding_cost_mmrc = bitrate
                        k_equiv = bitrate
                        best_k, best_bias = estimate_k_and_bias(d, M_rrsc, eps)
                    elif vary == 'bitrate':
                        bitrate = indices[step]
                        M_rrsc = int(2 ** bitrate)  # number of codewords for LDP-RD
                        coding_cost_mmrc = bitrate
                        k_equiv = bitrate
                        best_k, best_bias = estimate_k_and_bias(d, M_rrsc, eps)

                    bitrate_dict["sqkr"] = k_equiv
                    bitrate_dict["mmrc"] = coding_cost_mmrc
                    bitrate_dict["rrsc"] = int(np.ceil(np.log2(M_rrsc)))

                print(f"--------------\n d= {d}, n = {n}, eps = {eps}, bitrate = {bitrate}")
                for itr in range(num_itr):
                    if vary in  ["users", "d"]:
                        # Generate data matrix
                        X = np.zeros((d, n))
                        for j in range(int(n / 2)):
                            v_1 = np.concatenate([np.random.normal(10, 1, int(d))])
                            v_2 = np.concatenate([np.random.normal(1, 1, int(d))])
                            X[:, j] = v_1 / np.linalg.norm(v_1)
                            X[:, j + int(n / 2)] = v_2 / np.linalg.norm(v_2)
                    else:
                        X = input[itr]

                    # SQKR
                    # Generate a random tight frame satisfying UP
                    N = 2 ** int(np.ceil(np.log2(d)) + 1)
                    U = ortho_group.rvs(dim=N).T[:, 0:d]
                    start_time = time.time()
                    [q_quantize, q_sampling, q_perturb] = kashin_encode(U, X, k_equiv, eps)
                    X_hat = kashin_decode(U, k_equiv, eps, q_perturb)
                    mse = np.linalg.norm(np.mean(X, axis=1).reshape(-1, 1) - X_hat) ** 2
                    run_mse["sqkr"][step][itr] = mse
                    mse_dict["sqkr"][step] = mse_dict["sqkr"][step] + mse * 1 / num_itr
                    if itr == 0:
                        print("--- %.3f seconds for SQKR ---" % (time.time() - start_time))

                    # MMRC
                    start_time = time.time()
                    eta = eps / 2.0
                    x_miracle = np.zeros((d, n))
                    c1, c2, m, gamma = get_parameters_unbiased_miracle(
                        eps, d, 2**coding_cost_mmrc, budget
                    )
                    for i in range(n):
                        _, _, pi = mmrc_encoder(i + itr * n, X[:, i], 2**coding_cost_mmrc, c1, c2, gamma)
                        pi_all = modify_pi(pi, eta, eps, c1 / (np.exp(eps / 2)))
                        k = np.random.choice(2**coding_cost_mmrc, 1, p=pi_all[-1])[0]
                        z_k = mmrc_decoder(i + itr * n, k, d, 2**coding_cost_mmrc)
                        x_miracle[:, i] = z_k / m
                    x_miracle = np.mean(x_miracle, axis=1, keepdims=True)
                    mse = np.linalg.norm(np.mean(X, axis=1, keepdims=True) - x_miracle) ** 2
                    run_mse["mmrc"][step][itr] = mse
                    mse_dict["mmrc"][step] = mse_dict["mmrc"][step] + mse * 1 / num_itr
                    if itr == 0:
                        print("--- %.3f seconds for MMRC ---" % (time.time() - start_time))

                    # privUnitG
                    if vary == 'bitrate' and step > 0:
                        mse_dict["privunit"][step] = mse_dict["privunit"][0]
                        run_mse["privunit"][step][itr] = run_mse["privunit"][0][itr]
                    else:
                        start_time = time.time()
                        X_perturb = PrivUnitG_n_users(X, eps)
                        X_hat = np.mean(np.array(X_perturb), axis=1).reshape(-1, 1)
                        mse = np.linalg.norm(np.mean(X, axis=1).reshape(-1, 1) - X_hat) ** 2
                        run_mse["privunit"][step][itr] = mse
                        mse_dict["privunit"][step] = mse_dict["privunit"][step] + mse * 1 / num_itr
                        if itr == 0:
                            print("--- %.3f seconds for PrivUnit ---" % (time.time() - start_time))

                    # RRSC
                    start_time = time.time()
                    (d, n) = X.shape
                    X_perturb = lossy_DP_top1(X, eps, M_rrsc, best_k, itr) * best_bias
                    X_hat =  np.mean(X_perturb, axis=1).reshape(-1, 1)

                    mse = np.linalg.norm(np.mean(X, axis=1).reshape(-1, 1) - X_hat) ** 2
                    run_mse["rrsc"][step][itr] = mse
                    mse_dict["rrsc"][step] = mse_dict["rrsc"][step] + mse * 1 / num_itr
                    if itr == 0:
                        print("--- %.3f seconds for RRSC ---" % (time.time() - start_time))

            print("--------------")

            data = {
                "d": d,
                "num_user": n,
                "eps": eps,
                "mse_dict": mse_dict,
                "run_mse": run_mse,
                "indices": indices,
                "varied_param": vary,
            }

            para = f"sweep_{vary}_d_{d}_eps_{eps}_n_{n}_M_{M_rrsc}.mat"
            folder_name = "Data"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            fname = os.path.join(folder_name, para)
            io.savemat(fname, data)
