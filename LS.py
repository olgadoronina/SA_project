import numpy as np
import logging
from scipy.linalg import svd

import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt

folder = './plots/'
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rc('text', usetex=True)
mpl.rcParams['axes.labelsize'] = mpl.rcParams['font.size']
mpl.rcParams['axes.titlesize'] = 1.5 * mpl.rcParams['font.size']
mpl.rcParams['legend.fontsize'] = mpl.rcParams['font.size']
mpl.rcParams['xtick.labelsize'] = mpl.rcParams['font.size']
mpl.rcParams['ytick.labelsize'] = mpl.rcParams['font.size']
# plt.rcParams['savefig.dpi'] = 2 * plt.rcParams['savefig.dpi']
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
# mpl.rcParams['legend.frameon'] = False
# plt.rcParams['legend.loc'] = 'center left'
mpl.rcParams['axes.linewidth'] = 1

np.set_printoptions(precision=4, suppress=True)
N_params = 10
N_elem_in_tensor = 6
flatten_size = 256**3

calculate = 1

logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)

if calculate:
    logging.info('Loading sigma from files .npz')
    sigma_dict = np.load('./data_input/sigma_true.npz')
    sigma_true = np.empty(N_elem_in_tensor*flatten_size, dtype=np.float32)
    for ind, key in enumerate(['uu', 'uv', 'uw', 'vv', 'vw', 'ww']):
        start = flatten_size*ind
        sigma_true[start:start+flatten_size] = sigma_dict[key].flatten()
    print('sigma shape ', sigma_true.shape)

    logging.info('Loading Tensors from files .npz')
    Tensor = np.empty((N_params, N_elem_in_tensor*flatten_size))
    for i in range(N_params):
        tensor_dict = np.load('./data_input/Tensor'+str(i)+'.npz')
        for ind, key in enumerate(['uu', 'uv', 'uw', 'vv', 'vw', 'ww']):
            start = flatten_size * ind
            Tensor[i][start:start + flatten_size] = tensor_dict[key].flatten()
    Tensor = Tensor.T

    print('Tensor shape ', Tensor.shape)

    rank = np.linalg.matrix_rank(Tensor)
    print('Rank = ', rank)
    C = np.empty((N_elem_in_tensor, N_params))
    S = []
    fig, axes = plt.subplots(2, ncols=3, sharey=True, figsize=(10, 6))


    # 10 params
    # A = np.linalg.inv(Tensor.T @ Tensor) @ Tensor.T
    # c_10 = A @ sigma_true
    # np.savetxt(folder+'coef_10.txt', c_10)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes()
    # 60 params
    for ind in range(N_elem_in_tensor):
        start = flatten_size * ind
        end = start + flatten_size
        tensor = Tensor[start:end]
        sigma = sigma_true[start:end]
        A = np.linalg.inv(tensor.T @ tensor) @ tensor.T
        # rank = np.linalg.matrix_rank(A)
        # # print('rank', rank)
        # print(ind)

        # c = A @ sigma
        # print('C ', c)
        # C[ind, :] = c

        # # Covariance matrix
        # Cov_mL2 = np.linalg.inv(tensor.T @ tensor)
        # print(Cov_mL2)

        # SVD decomposition
        s = np.linalg.svd(tensor, full_matrices=False, compute_uv=False)  # Note: V = v.t

        S.append(s)

        print('diag(S) = {}'.format(s))



        ax.plot(s, 'o', label=str(ind+1))
    plt.title('Spectrum of singular values')
    plt.xlabel('i')
    plt.ylabel('$s_i$')
    plt.legend()
    fig.savefig(folder + 's_value')


    #     G_dagger = np.linalg.pinv(tensor)
    #     C[ind, :] = G_dagger @ sigma
    #     print('C ', C[ind])
    #
    #     # Calculate model resolution matrix
    #     Rm = G_dagger.dot(tensor)
    #     print('diag(Rm) ', np.diag(Rm))
    #     im = axes.flat[ind].imshow(Rm.T)
    # fig.colorbar(im)
    # fig.savefig('./plots/Rm')

    np.savetxt(folder+'coef.txt', C)

else:
    C = np.loadtxt(folder+'coef.txt')
    c_10 = np.loadtxt(folder+'coef_10.txt')

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes()

    for i in range(len(C)):
        ax.plot(C[i], 'o', label=str(i+1))
    ax.plot(c_10, 's', label='10 params')

    plt.title('Coefficients')
    plt.xlabel(r'$i$')
    plt.ylabel(r'$C_i$')
    plt.legend()
    fig.savefig(folder + 'Coef')