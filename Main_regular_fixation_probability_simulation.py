import networkx as nx
import random
import numpy as np
import pickle
# import matplotlib.pyplot as plt
from numba import jit
import os
import multiprocessing
# from multiprocessing import Process, Manager
import functools
import time
import math
import scipy.io as scio


# Numerical simulations of the fixation probability of cooperation (rho_c) given
# a network adjacent matrix and individual update rates

@jit(nopython=True)
def single_game(state_array, payoff_array, game_matrix, edge_array, deg_array):
    """
    each individual plays a single game with all its neighbors, and return the accumulated payoff in this game
    Note: all the inputs variables are numpy array to enable the jit acceleration
    :param state_array: the state of each individual
    :param payoff_array: the payoff of each individual
    :param game_matrix: game matrix for the donation game
    :param edge_array: the numpy array for edges
    :return: average payoff for each individual
    """

    num_edges = edge_array.shape[0]  # Precompute edge array size
    for i in range(num_edges):
        nodex = edge_array[i, 0]  # iterate each edge on the network
        nodey = edge_array[i, 1]
        # individuals on both ends of the edge play games and accumulate payoffs
        payoff_array[nodex] += game_matrix[state_array[nodex]][state_array[nodey]]
        payoff_array[nodey] += game_matrix[state_array[nodey]][state_array[nodex]]

    payoff_array = payoff_array / deg_array
    return payoff_array  # average payoff for each individual


@jit(nopython=True)
def strategy_update(state_array, payoff_array, nbr_array, deg_array, w):
    """

    :param state_array: the state of each individual
    :param payoff_array: the payoff of each individual
    :param deg_array: the numpy array for nodes' degree
    :return: the state array after strategy update

    """

    update_node = np.random.randint(0, N)  # choose the update node with probability proportional to its update rate
    nbrs_num = deg_array[update_node]  # the number of neighbors of this update node
    nbrs_array = nbr_array[update_node][:nbrs_num]  # numpy array of the neighbors of the update node

    fitness_group = 1 + w * payoff_array[nbrs_array]  # fitness of the update node's neighbors
    prob_array = fitness_group / np.sum(fitness_group)  # probability for the update node imitating strategy from one of its neighbors
    state_array[update_node] = state_array[rand_pick_list(nbrs_array, prob_array)]  # strategy update

    return state_array


@jit(nopython=True)
def evolution(game_matrix, edge_array, nbr_array, deg_array, nodesnum, w):
    """
    the evolutionary process of cooperation starting from a single cooperator and end when the population reaches
     full cooperation or full defection
    :param game_matrix: game matrix for the donation game
    :param deg_array: numpy array of degree for each node
    :param nodesnum: network size
    :param w: selection intensity
    :return: frequency of cooperators in a realization of evolutionary process
    """

    payoff_array = np.zeros(nodesnum, dtype=np.float_)  # array of individual payoff
    state_array = np.zeros(nodesnum, dtype=np.int_)  # array of individual state: 1-->cooperation; 0-->defection
    mut_ini = np.random.choice(nodesnum)
    state_array[mut_ini] = 1  # randomly choose a single node as mutation
    coord = np.sum(state_array)

    while coord != 0 and coord != nodesnum:
        payoff_array = single_game(state_array, payoff_array, game_matrix, edge_array, deg_array)  # all individuals plays around of game
        state_array = strategy_update(state_array, payoff_array, nbr_array, deg_array, w)  # an individual updates its strategy according to its update rate
        payoff_array[:] = 0  # clear the payoff for each individual for the new round of game
        coord = np.sum(state_array)  # count the number of cooperators

    return coord / nodesnum  # return the frequency of cooperators


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
        return 0.0  # or some other error handling

    return det_tran / det_tran_norm


@jit(nopython=True)
def process(core, f, g, p, q, b, c, edge_array, nbr_array, deg_array, nodesnum):
    """
    Repeat the simulations to compute the fixation probability of cooperation by calculate the fraction of simulations
    in which the population reaches full cooperation out of realizations reaching absorption.
    :param b: the benefit in the donation game
    :param deg_array: numpy array of degree for each node
    :param nodesnum: network size
    :return: fixation probability of cooperation
    """

    w = 0.01

    game_matrix = np.zeros((2, 2))  # game matrix of the donation game in the main text
    game_matrix[1][1] = expected_payoff(p, q, p, q, b, c)  # P defect--defect
    game_matrix[1][0] = expected_payoff(p, q, f, g, b, c)  # T defect-cooperate
    game_matrix[0][1] = expected_payoff(f, g, p, q, b, c)  # S cooperate-defect
    game_matrix[0][0] = expected_payoff(f, g, f, g, b, c)  # R cooperate-cooperate

    repeat_time = int(1e6)  # total number of realizations
    repeat_array = np.zeros(repeat_time)

    for rep in range(repeat_time):
        freq_c = evolution(game_matrix, edge_array, nbr_array, deg_array, nodesnum, w)  # frequency of cooperators in a realization of evolutionary process
        repeat_array[rep] = freq_c

    return np.sum(repeat_array == 1) / (np.sum(repeat_array == 1) + np.sum(repeat_array == 0))  # only count processes which reach absorption state


@jit(nopython=True)
def rand_pick_list(pick_list, prob_list):
    """
    choose an element according to a given probability distribution, which performs the same as "numpy.random.choice",
    since the function "numpy.random.choice" can not run under numba jit acceleration
    :param pick_list: the element array
    :param prob_list: the array of probability distribution
    :return: the chosen element
    """
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(pick_list, prob_list):
        cumulative_probability += item_probability
        if x <= cumulative_probability:
            break
    return item


if __name__ == "__main__":
    N=200
    k=2
    c=1
    epsilon=0.001
    f=1-epsilon
    g=1-epsilon
    p=epsilon
    q=epsilon

    while True:
        G = nx.random_regular_graph(k, N)
        if nx.is_connected(G) and len(list(nx.selfloop_edges(G))) == 0 and G.number_of_edges() == (N * k)//2:
            break

    # 1. Convert the edge list to a NumPy array
    edge_list = list(G.edges())
    edge_array = np.array(edge_list)

    # 2. Convert the neighbors of each node to a NumPy array
    nbrs_dict = nx.to_dict_of_lists(G)  # Get neighbors as a dictionary
    max_neighbors = max(len(v) for v in nbrs_dict.values())  # Find the max number of neighbors any node has

    # Initialize a NumPy array to store neighbors (-1 means no neighbor for shorter lists)
    nbr_array = np.full((N, max_neighbors), -1)
    deg_array = np.zeros(N, int)

    for node, neighbors in nbrs_dict.items():
        nbr_array[node, :len(neighbors)] = neighbors
        deg_array[node] = len(neighbors)

    b_array = [1.6, 1.8, 2, 2.2, 2.4, 2.6]  # array of benefit b in donation game, example in Fig. 2b
    cpu_cores_num = 10  # 10-cpu core, this number should match to cpu-core running this programme
    rhoc_array = []

    for b_para in b_array:
        core_list = np.arange(cpu_cores_num)

        # start parallel computing the fixation probability of cooperation rho_c
        pool = multiprocessing.Pool()
        t1 = time.time()

        pt = functools.partial(process, f=f, g=g, p=p, q=q, b=b_para, c=c, edge_array=edge_array, nbr_array=nbr_array,
                               deg_array=deg_array, nodesnum=N)  # perform the process function on each core
        rho_c_list = pool.map(pt, core_list)  # return the fixation probability rho_c computed on each cpu core
        rho_c = sum(rho_c_list) / len(rho_c_list)  # calculate the average rho_c
        rhoc_array.append(rho_c)
        pool.close()
        pool.join()
        t2 = time.time()
        print("b=" + str(b_para) + ",", "rho_c=" + str(rho_c))
        print("Total time:" + (t2 - t1).__str__())

    file = "average_regular_n200_k2_alld_with_error_new_w.mat"

    # output: array of the benefit b, and array of the corresponding fixation probability of cooperation (rho_c)
    # note: here the cost c=1 (see the game matrix), therefore b/c is equal to b
    scio.savemat(file, {'b_array_alld': b_array,
                        'rhoc_array_alld': rhoc_array})


