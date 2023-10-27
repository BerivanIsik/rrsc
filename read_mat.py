import scipy.io
import os

if not os.path.exists('Data_paper'):
    os.makedirs('Data_paper')
fname = os.path.join('Data_paper', 'd_100_eps_2_n_100_M_4.mat')
mat = scipy.io.loadmat(fname)
print(mat['mse_dict'][0][0][0])
print(mat['mse_dict'][0][0][1])
print(mat['mse_dict'][0][0][2])
print(mat['mse_dict'][0][0][3])