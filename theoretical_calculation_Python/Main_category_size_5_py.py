import os
import numpy as np
import scipy.io
import networkx as nx
import matplotlib.pyplot as plt
from network_calc import f_network_category

if not os.path.exists('n_5.mat') or not os.path.exists('n_5_coords.mat'):
    print("Error: 'n_5.mat' or 'n_5_coords.mat' not found.")
    exit()

# Load predefined adjacency matrices (graph_1, graph_2, ..., graph_21)
data = scipy.io.loadmat('n_5.mat')
coords_data = scipy.io.loadmat('n_5_coords.mat')

# Initialize category results for 21 networks
cate = np.zeros(21)
for i in range(1, 22):
    G_mat = data[f'graph_{i}']
    # Compute the category (1: strongly prosocial, 0: weakly prosocial, -1: antisocial)
    cate[i - 1] = f_network_category(G_mat)

print("cate = \n", cate)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create figure for visualizing all 21 networks
fig, axes = plt.subplots(3, 7, figsize=(14 / 2.54, 6 / 2.54), gridspec_kw={'wspace': 0.05, 'hspace': 0.1})
# Create a tight 3-by-7 subplot layout
axes = axes.flatten()

color_prosocial = [131 / 255, 203 / 255, 172 / 255]
color_weakly = [147 / 255, 181 / 255, 207 / 255]
color_antisocial = [240 / 255, 161 / 255, 168 / 255]
edge_color = [181 / 255, 181 / 255, 182 / 255]

# Plot each network in the corresponding subplot
for i in range(21):
    ax = axes[i]

    # Retrieve adjacency matrix by variable name
    G_mat = data[f'graph_{i + 1}']

    # Ensure matrix is numeric and convert to graph object
    G = nx.from_numpy_array(G_mat)

    mat_coords = coords_data[f'graph_{i + 1}']
    pos = {node_idx: (mat_coords[0, node_idx], mat_coords[1, node_idx]) for node_idx in range(len(G))}

    # Visualize the network according to its category
    if cate[i] == 1:
        # Strongly prosocial structure
        node_color = color_prosocial
    elif cate[i] == 0:
        # Weakly prosocial structure
        node_color = color_weakly
    else:
        # Antisocial structure
        node_color = color_antisocial

    edges = nx.draw_networkx_edges(G, pos, ax=ax, edge_color=[edge_color], alpha=0.5, width=0.6)
    if edges:
        edges.set_zorder(1)

    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[node_color], node_size=12, linewidths=0)
    if nodes:
        nodes.set_zorder(2)

    ax.axis('on')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#333333')
        spine.set_linewidth(0.5)

plt.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.13)
plt.show()