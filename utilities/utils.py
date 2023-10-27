import numpy as np
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.stats import norm


def display_results(mse_dict, step):
    if step is not None:
        for method, mse in mse_dict.items():
            print(f"{method} mse: {mse[step]}")
    else:
        # If None, print MSEs of all steps
        for method, mse in mse_dict.items():
            print(f"{method} mse:")
            print(mse)

def plot_vary_users_results(mse_dict, indices, eps, bitrate_dict, filename):
    plt.figure(figsize=(6, 4.5))
    for method, mse in mse_dict.items():
        plt.plot(np.array(indices), mse, label=f"{method}({bitrate_dict[method]} bits)")
        plt.legend()
    plt.show()
    plt.xlabel(r"n", fontsize=18)
    plt.ylabel(r"$\ell_2$ error", fontsize=18)
    title = "Error vs. Num. of Users ($\epsilon$=" + str(eps) + ")"
    plt.title(title, fontsize=18)
    plt.savefig(filename)

def plot_vary_bitrate_results(mse_dict, indices, eps, bitrate_dict, filename):
    plt.figure(figsize=(6, 4.5))
    for method, mse in mse_dict.items():
        plt.plot(np.array(indices), mse, label=f"{method}")
        plt.legend()
    plt.show()
    plt.xlabel(r"bitrate", fontsize=18)
    plt.ylabel(r"$\ell_2$ error", fontsize=18)
    title = "Error vs. Bitrate ($\epsilon$=" + str(eps) + ")"
    plt.title(title, fontsize=18)
    plt.savefig(filename + ".png")
