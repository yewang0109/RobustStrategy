# Core network computation functions

import math
import itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def get_comb_dicts(n):
    dict_2 = {comb: idx for idx, comb in enumerate(itertools.combinations(range(n), 2))}
    dict_3 = {comb: idx for idx, comb in enumerate(itertools.combinations(range(n), 3))}
    return dict_2, dict_3


def f_gen_trans_mat(madj, n):
    trans_mat = madj.astype(float)
    row_sums = trans_mat.sum(axis=1)
    return trans_mat / row_sums[:, np.newaxis]


def f_cal_remeet_time_two(trans_mat, n, dict_2):
    dict_size = math.comb(n, 2)

    id_x_arr, id_y_arr, val_arr = [], [], []
    b_arr = np.full(dict_size, -0.5)

    for i, j in itertools.combinations(range(n), 2):
        index_idx = dict_2[(i, j)]
        id_x_arr.append(index_idx)
        id_y_arr.append(index_idx)
        val_arr.append(-1.0)

        for k in range(n):
            if k == j or trans_mat[i, k] == 0: continue
            seq = tuple(sorted([k, j]))
            index_idy = dict_2[seq]
            id_x_arr.append(index_idx)
            id_y_arr.append(index_idy)
            val_arr.append(trans_mat[i, k] / 2.0)

        for k in range(n):
            if k == i or trans_mat[j, k] == 0: continue
            seq = tuple(sorted([i, k]))
            index_idy = dict_2[seq]
            id_x_arr.append(index_idx)
            id_y_arr.append(index_idy)
            val_arr.append(trans_mat[j, k] / 2.0)

    adj_mat = -sp.coo_matrix((val_arr, (id_x_arr, id_y_arr)), shape=(dict_size, dict_size)).tocsr()
    b_arr = -b_arr

    retime, _ = spla.bicgstab(adj_mat, b_arr, tol=1e-6, maxiter=10 ** 5)
    return retime


def f_cal_remeet_time_three(trans_mat, retime2, n, dict_2, dict_3):
    dict_size = math.comb(n, 3)
    id_x_arr, id_y_arr, val_arr = [], [], []
    b_arr = np.full(dict_size, -1 / 3)

    for i, j, k in itertools.combinations(range(n), 3):
        index_idx = dict_3[(i, j, k)]
        id_x_arr.append(index_idx)
        id_y_arr.append(index_idx)
        val_arr.append(-1.0)

        def process_transitions(node, other1, other2):
            for l in range(n):
                if trans_mat[node, l] == 0: continue
                seq = tuple(sorted(set([l, other1, other2])))
                length = len(seq)
                if length == 1:
                    continue
                elif length == 2:
                    idx = dict_2[seq]
                    val = retime2[idx] * trans_mat[node, l] / 3.0
                    b_arr[index_idx] -= val
                else:
                    idy = dict_3[seq]
                    id_x_arr.append(index_idx)
                    id_y_arr.append(idy)
                    val_arr.append(trans_mat[node, l] / 3.0)

        process_transitions(i, j, k)
        process_transitions(j, i, k)
        process_transitions(k, i, j)

    adj_mat = -sp.coo_matrix((val_arr, (id_x_arr, id_y_arr)), shape=(dict_size, dict_size)).tocsr()
    b_arr = -b_arr
    retime, _ = spla.bicgstab(adj_mat, b_arr, tol=1e-6, maxiter=10 ** 5)
    return retime


def f_reshape_retime(retime2, retime3, n, dict_2, dict_3):
    retime2_mat = np.zeros((n, n))
    retime3_mat = np.zeros((n, n, n))

    for i, j in itertools.combinations(range(n), 2):
        val = retime2[dict_2[(i, j)]]
        retime2_mat[i, j] = val
        retime2_mat[j, i] = val

    for i, j, k in itertools.combinations(range(n), 3):
        val = retime3[dict_3[(i, j, k)]]
        for perm in itertools.permutations([i, j, k]):
            retime3_mat[perm] = val

    for i, j in itertools.combinations(range(n), 2):
        val = retime2_mat[i, j]
        retime3_mat[i, i, j] = val
        retime3_mat[i, j, i] = val
        retime3_mat[j, i, i] = val
        retime3_mat[j, j, i] = val
        retime3_mat[j, i, j] = val
        retime3_mat[i, j, j] = val

    return retime2_mat, retime3_mat


def f_calculation_tau(G_mat):
    n = G_mat.shape[0]
    dict_2, dict_3 = get_comb_dicts(n)

    trans_mat = f_gen_trans_mat(G_mat, n)
    retime2 = f_cal_remeet_time_two(trans_mat, n, dict_2)
    retime3 = f_cal_remeet_time_three(trans_mat, retime2, n, dict_2, dict_3)
    retime2_mat, retime3_mat = f_reshape_retime(retime2, retime3, n, dict_2, dict_3)

    rowSums = G_mat.sum(axis=1)
    G_1 = G_mat / rowSums[:, np.newaxis]
    G_2 = G_1 @ G_1
    G_3 = G_1 @ G_1 @ G_1

    a = np.eye(n)
    sum_rowSums = np.sum(rowSums)

    cont1, cont2, cont3 = 0.0, 0.0, 0.0

    for i in range(n):
        coef = rowSums[i] / sum_rowSums
        for j in range(n):
            cont1 += coef * (G_2[i, j] - a[i, j]) * retime2_mat[i, j]
            cont2 += coef * (G_3[i, j] - G_1[i, j]) * retime2_mat[i, j]
            for y in range(n):
                cont3 += coef * (G_2[i, j] - a[i, j]) * G_1[j, y] * retime3_mat[i, j, y]

    return cont1, cont2, cont3


def f_network_category(G_mat):
    tau_1, tau_2, tau_3 = f_calculation_tau(G_mat)
    tau_1, tau_2, tau_3 = round(tau_1, 4), round(tau_2, 4), round(tau_3, 4)

    if tau_2 <= 0:
        return -1

    if tau_3 > 0:
        if tau_2 >= tau_3:
            return 1
        else:
            critical_ep = (2 * tau_2 - tau_3) / (2 * (tau_2 - tau_3))
    elif tau_3 < 0:
        critical_ep = -tau_3 / (2 * (tau_2 - tau_3))
    else:
        return np.nan

    critical_ep = min(max(critical_ep, 0), 0.5)

    if critical_ep == 0:
        return 1
    elif critical_ep == 0.5:
        return -1
    else:
        return 0