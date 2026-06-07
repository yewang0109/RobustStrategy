import numpy as np
from network_calc import f_calculation_tau

# Construct a fan graph with n branches
n = 25
G = np.zeros((2 * n + 1, 2 * n + 1))
G[0, 1:2 * n + 1] = 1
G[1:2 * n + 1, 0] = 1
for branch in range(1, n + 1):
    G[2 * branch - 1, 2 * branch] = 1
    G[2 * branch, 2 * branch - 1] = 1

# Compute tau_1, tau_2, and tau_3
tau_1, tau_2, tau_3 = f_calculation_tau(G)
tau_1 = round(tau_1, 4)
tau_2 = round(tau_2, 4)
tau_3 = round(tau_3, 4)

# Parameters
e = 0.001  # error rate
delta = 0.01  # selection strength
b = 25  # benefit
c = 1  # cost

# Transition matrix initialization
trans = np.zeros((4, 4))
N = 2 * n + 1  # population size
cont1 = -1  # index counter for states

for i in range(1, 3):
    p = (i - 1) * e + (2 - i) * (1 - e)
    for j in range(1, 3):
        q = (j - 1) * e + (2 - j) * (1 - e)
        s11 = q * (c - b) / (p - q - 1)
        cont1 = cont1 + 1
        cont2 = -1
        for h in range(1, 3):
            f = (h - 1) * e + (2 - h) * (1 - e)
            for l in range(1, 3):
                cont2 = cont2 + 1
                g = (l - 1) * e + (2 - l) * (1 - e)
                s10 = (c * g * (p - q) - b * q * (f - g) + c * q - b * g) / ((f - g) * (p - q) - 1)
                s01 = (c * q * (f - g) - b * g * (p - q) + c * g - b * q) / ((f - g) * (p - q) - 1)
                s00 = g * (c - b) / (f - g - 1)

                # transition probability under weak selection
                trans[cont2, cont1] = 1 / N + delta / N * \
                                      ((s10 - s00) * tau_1 + (s01 - s00) * tau_2 + (s11 + s00 - s01 - s10) * tau_3)

for i in range(4):
    trans[i, i] = 1 - np.sum(trans[i, :]) + trans[i, i]

print('transition_matrix =')
for i in range(trans.shape[0]):
    row_str = ""
    for j in range(trans.shape[1]):
        row_str += f"{trans[i, j]:11.5f}"
    print(row_str)