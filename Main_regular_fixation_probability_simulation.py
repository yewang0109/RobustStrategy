import networkx as nx
import random
import numpy as np
import pickle
from numba import jit
import os
import multiprocessing
import functools
import time
import math
import scipy.io as scio


# Numerical simulations of the fixation probability given a network adjacent matrix
@jit(nopython=True)
def single_game(state_array, payoff_array, game_matrix, edge_array, deg_array):
    """Calculates and returns the average payoff for each individual."""
    num_edges = edge_array.shape[0]
    for i in range(num_edges):
        nodex = edge_array[i, 0]
        nodey = edge_array[i, 1]

        payoff_array[nodex] += game_matrix[state_array[nodex]][state_array[nodey]]
        payoff_array[nodey] += game_matrix[state_array[nodey]][state_array[nodex]]

    payoff_array = payoff_array / deg_array
    return payoff_array


@jit(nopython=True)
def strategy_update(state_array, payoff_array, nbr_array, deg_array, w):
    """Updates the state array based on fitness."""
    update_node = np.random.randint(0, N)
    nbrs_num = deg_array[update_node]
    nbrs_array = nbr_array[update_node][:nbrs_num]

    fitness_group = 1 + w * payoff_array[nbrs_array]
    prob_array = fitness_group / np.sum(fitness_group)
    state_array[update_node] = state_array[rand_pick_list(nbrs_array, prob_array)]

    return state_array


@jit(nopython=True)
def evolution(game_matrix, edge_array, nbr_array, deg_array, nodesnum, w):
    """Evolutionary process starting from a single mutation until absorption."""
    payoff_array = np.zeros(nodesnum, dtype=np.float_)
    state_array = np.zeros(nodesnum, dtype=np.int_)
    mut_ini = np.random.choice(nodesnum)
    state_array[mut_ini] = 1
    coord = np.sum(state_array)

    while coord != 0 and coord != nodesnum:
        payoff_array = single_game(state_array, payoff_array, game_matrix, edge_array, deg_array)
        state_array = strategy_update(state_array, payoff_array, nbr_array, deg_array, w)
        payoff_array[:] = 0
        coord = np.sum(state_array)

    return coord / nodesnum


@jit(nopython=True)
def expected_payoff(p, q, f, g, b, c):
    tran_matrix = np.array([[p * f - 1, p - 1, f - 1, b - c],
                            [q * f, q - 1, f, -c],
                            [p * g, p, g - 1, b],
                            [q * g, q, g, 0]], dtype=np.float64)
    tran_matrix_norm = np.array([[p * f - 1, p - 1, f - 1, 1],
                                 [q * f, q - 1, f, 1],
                                 [p * g, p, g - 1, 1],
                                 [q * g, q, g, 1]], dtype=np.float64)

    det_tran = np.linalg.det(tran_matrix)
    det_tran_norm = np.linalg.det(tran_matrix_norm)

    if det_tran == 0 or det_tran_norm == 0:
        return 0.0

    return det_tran / det_tran_norm


@jit(nopython=True)
def process(core, f, g, p, q, b, c, edge_array, nbr_array, deg_array, nodesnum):
    """Computes the fixation probability over multiple realizations."""
    w = 0.01

    game_matrix = np.zeros((2, 2))
    game_matrix[1][1] = expected_payoff(p, q, p, q, b, c)
    game_matrix[1][0] = expected_payoff(p, q, f, g, b, c)
    game_matrix[0][1] = expected_payoff(f, g, p, q, b, c)
    game_matrix[0][0] = expected_payoff(f, g, f, g, b, c)

    repeat_time = int(1e6)
    repeat_array = np.zeros(repeat_time)

    for rep in range(repeat_time):
        freq_c = evolution(game_matrix, edge_array, nbr_array, deg_array, nodesnum, w)
        repeat_array[rep] = freq_c

    return np.sum(repeat_array == 1) / (np.sum(repeat_array == 1) + np.sum(repeat_array == 0))


@jit(nopython=True)
def rand_pick_list(pick_list, prob_list):
    """Chooses an element according to a given probability distribution."""
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(pick_list, prob_list):
        cumulative_probability += item_probability
        if x <= cumulative_probability:
            break
    return item


if __name__ == "__main__":
    N = 200
    k = 2
    c = 1
    epsilon = 0.001
    f = 1 - epsilon
    g = 1 - epsilon
    p = epsilon
    q = epsilon

    while True:
        G = nx.random_regular_graph(k, N)
        if nx.is_connected(G) and len(list(nx.selfloop_edges(G))) == 0 and G.number_of_edges() == (N * k) // 2:
            break

    edge_list = list(G.edges())
    edge_array = np.array(edge_list)

    nbrs_dict = nx.to_dict_of_lists(G)
    max_neighbors = max(len(v) for v in nbrs_dict.values())

    nbr_array = np.full((N, max_neighbors), -1)
    deg_array = np.zeros(N, int)

    for node, neighbors in nbrs_dict.items():
        nbr_array[node, :len(neighbors)] = neighbors
        deg_array[node] = len(neighbors)

    b_array = [1.6, 1.8, 2, 2.2, 2.4, 2.6]
    cpu_cores_num = 10
    rhoc_array = []

    for b_para in b_array:
        core_list = np.arange(cpu_cores_num)

        pool = multiprocessing.Pool()
        t1 = time.time()

        pt = functools.partial(process, f=f, g=g, p=p, q=q, b=b_para, c=c, edge_array=edge_array, nbr_array=nbr_array,
                               deg_array=deg_array, nodesnum=N)
        rho_c_list = pool.map(pt, core_list)
        rho_c = sum(rho_c_list) / len(rho_c_list)
        rhoc_array.append(rho_c)

        pool.close()
        pool.join()

        t2 = time.time()
        print("b=" + str(b_para) + ",", "rho_c=" + str(rho_c))
        print("Total time:" + (t2 - t1).__str__())

    file = "average_regular_n200_k2_alld_with_error_new_w.mat"

    scio.savemat(file, {'b_array_alld': b_array,
                        'rhoc_array_alld': rhoc_array})