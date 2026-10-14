#!/usr/bin/env python3
"""Finite-N memory-one invasion robustness on the three paper networks.

Run the complete calculation (all 16 residents; 16 corners and 9**4 grid):
    python theory.py
Small example:
    python theory.py --networks cycle --epsilon 0.1 --residents 1111 --invaders corners
Choose an output directory with --output-dir (default: results/theory).

Only NumPy and SciPy are required. No plot or statistical-analysis package is
used. N=50; death-birth updating; uniform placement of a single mutant;
weak-selection first-order criterion; neighbor-averaged donation-game payoffs.
The effective cooperation coordinates are ordered CC, CD, DC, DD. They already
include execution error: corners are epsilon or 1-epsilon; the grid uses nine
evenly spaced values in this interval on each coordinate. No second error
transformation is applied. Cost c=1 and b/c>1 are used for interval output.

For a mutant M and resident R, invasion is not favored at first order when
 F = tau1*(s_MR-s_RR) + tau2*(s_RM-s_RR)
     + tau3*(s_MM-s_MR-s_RM+s_RR) <= 0.
Stationary payoffs are linear in b,c, so F/c=alpha*(b/c)+beta. The algorithm
intersects these half-lines analytically; it does not sample b/c. Finite closed
endpoints are first-order neutral, not a strict-ESS claim. Checking a finite
grid does not establish robustness on the continuous strategy space.

This file consolidates network_calc.py, allc_robustness_intervals.py and
corner16_robustness_intervals.py from the paper's reference implementation.
Its two-/three-lineage equations and time normalization originate in
RobustStrategy/theoretical_calculation_Python/network_calc.py. Fixed graph
hashes below identify exactly the paper networks. Stationary solves are batched
for speed; the transition matrix and invasion inequalities are unchanged.

Public API: make_graphs(), network_coefficients(adjacency),
stationary_distribution(player, opponent), cooperation_rates(player, opponent),
memory_one_payoff(player, opponent, b, c=1), intersect_constraints(...).
"""
import argparse
import csv
import hashlib
import inspect
import itertools
import json
import math
from itertools import product
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

N = 50

EPSILON_MARGIN = 1e-4

BITS = np.array(list(product((0, 1), repeat=4)), dtype=int)

NAMES = {0: 'ALLDe', 5: 'ATFTe', 8: 'Grime', 9: 'WSLSe', 10: 'TFTe', 15: 'ALLCe'}

NETWORK_NAMES = {'cycle': 'Cycle (k=2)', 'regular': 'Regular (k=4)',
                 'wellmixed': 'Well-mixed (k=49)'}

REGULAR_EDGES = (
    (0, 1), (0, 36), (0, 39), (0, 43), (1, 2), (1, 33), (1, 34), (2, 3),
    (2, 10), (2, 12), (3, 9), (3, 35), (3, 39), (4, 5), (4, 6), (4, 9),
    (4, 19), (5, 16), (5, 19), (5, 41), (6, 7), (6, 20), (6, 26), (7, 17),
    (7, 28), (7, 41), (8, 9), (8, 22), (8, 30), (8, 45), (9, 45), (10, 11),
    (10, 25), (10, 37), (11, 29), (11, 32), (11, 44), (12, 15), (12, 43), (12, 48),
    (13, 14), (13, 23), (13, 38), (13, 49), (14, 16), (14, 23), (14, 25), (15, 18),
    (15, 46), (15, 47), (16, 35), (16, 40), (17, 43), (17, 46), (17, 47), (18, 21),
    (18, 27), (18, 30), (19, 20), (19, 22), (20, 38), (20, 49), (21, 22), (21, 32),
    (21, 45), (22, 46), (23, 24), (23, 25), (24, 25), (24, 36), (24, 39), (26, 30),
    (26, 35), (26, 41), (27, 28), (27, 42), (27, 48), (28, 29), (28, 37), (29, 36),
    (29, 42), (30, 47), (31, 32), (31, 33), (31, 34), (31, 38), (32, 40), (33, 34),
    (33, 48), (34, 44), (35, 37), (36, 40), (37, 41), (38, 39), (40, 45), (42, 44),
    (42, 48), (43, 46), (44, 49), (47, 49),
)

GRAPH_HASHES = {
    'cycle': '97eaae746556db69e2f21a2686a50fae6c8939918606a2ee2afaa1c6d5e07845',
    'regular': '6ddc32d8950ef2c2a0a037e7a69e922c7e0cb39aefbc07d63635e807c21ad82a',
    'wellmixed': 'a102affa131a3db42734658a3622a64b147b16cdc77a53fa495222c521de12aa',
}

SOURCE_SHA256 = {'network_calc.py': '4d7a76d5e48740c1c7063bfa1c0674f6f695ce7d9a5ff6d86f736a8778afcaf7', 'allc_robustness_intervals.py': '66e7886e78b8efb47aa0b244fb3d530bac22f632f2410addc8cda37aa17a62f6', 'corner16_robustness_intervals.py': 'b6f64df8fe9d756ee37a58778266fe9e1a01c48dc05940d572a3f17b19d324e7', 'interval_output.py': 'b0b3643a7381d17fc78cddc3e47b5ea6523faa2bde9c5f3b3345dbf5753ef82e'}


def _solve_coalescence_system(matrix, rhs):
    """调用 SciPy 求解原工具包的方程，不更改方程或时间归一化。"""
    tolerance_name = 'rtol' if 'rtol' in inspect.signature(spla.bicgstab).parameters else 'tol'
    options = {tolerance_name: 1e-12, 'atol': 1e-12, 'maxiter': 100000}
    solution, info = spla.bicgstab(matrix, rhs, **options)
    residual = np.max(np.abs(matrix @ solution - rhs))
    if info != 0 or not np.isfinite(solution).all() or residual > 1e-8:
        raise ArithmeticError(f'Coalescence solve failed: info={info}, residual={residual:g}')
    return solution

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

    retime = _solve_coalescence_system(adj_mat, b_arr)
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
    retime = _solve_coalescence_system(adj_mat, b_arr)
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


def network_coefficients(adjacency):
    """Return (tau1,tau2,tau3) from the finite two-/three-lineage equations.

    Accepts a connected, symmetric, nonnegative adjacency matrix with no self
    loops and at least three vertices. Solves are checked for convergence and
    residual <= 1e-8; no large-N or pair approximation is used.
    """
    a=np.asarray(adjacency,dtype=float)
    if (a.ndim!=2 or a.shape[0]!=a.shape[1] or len(a)<3
            or not np.isfinite(a).all() or np.any(a<0)
            or not np.array_equal(a,a.T) or np.any(np.diag(a)!=0)):
        raise ValueError('Expected a finite symmetric nonnegative adjacency matrix without self loops')
    if sp.csgraph.connected_components(sp.csr_matrix(a),directed=False)[0]!=1:
        raise ValueError('The network must be connected')
    return tuple(float(t) for t in f_calculation_tau(a))


def stationary_distribution(player, opponent):
    """Stationary probabilities of CC,CD,DC,DD from the player's perspective.

    Inputs have final dimension 4 and may be broadcast batches. Strictly
    interior probabilities make the four-state chain irreducible and aperiodic.
    """
    p,q=np.broadcast_arrays(np.asarray(player,dtype=float),np.asarray(opponent,dtype=float))
    if (p.ndim<1 or p.shape[-1]!=4 or not np.isfinite(p).all()
            or not np.isfinite(q).all() or np.any((p<=0)|(p>=1))
            or np.any((q<=0)|(q>=1))):
        raise ValueError('Strategies must have four finite effective probabilities strictly between 0 and 1')
    q=q[..., [0,2,1,3]]  # Opponent CD and DC are reversed in the focal perspective.
    transition=np.stack((p*q,p*(1-q),(1-p)*q,(1-p)*(1-q)),axis=-1)
    equations=np.swapaxes(transition,-1,-2)-np.eye(4)
    equations[..., -1, :]=1
    rhs=np.zeros(p.shape)
    rhs[..., -1]=1
    stationary=np.linalg.solve(equations,rhs[...,None])[...,0]
    residual=np.max(np.abs((stationary[...,None,:]@transition)[...,0,:]-stationary))
    if not np.isfinite(stationary).all() or stationary.min() < -2e-12 or residual>1e-10:
        raise ArithmeticError('The four-state stationary solve failed')
    stationary=np.maximum(stationary,0)
    return stationary/stationary.sum(axis=-1,keepdims=True)


def cooperation_rates(player, opponent):
    """Return own and opponent cooperation probabilities (scalars or batches)."""
    v=stationary_distribution(player,opponent)
    return v[...,0]+v[...,1],v[...,0]+v[...,2]


def memory_one_payoff(player, opponent, b, c=1.):
    """Focal stationary donation-game payoff: b*C_opponent - c*C_focal."""
    if not np.isfinite(b) or not np.isfinite(c):
        raise ValueError('Benefit and cost must be finite scalars')
    own,other=cooperation_rates(player,opponent)
    return b*other-c*own


def bit_label(index):
    return ''.join(str(bit) for bit in BITS[index])

def strategy_name(index):
    return NAMES.get(index, bit_label(index))

def make_graphs():
    cycle = np.zeros((N, N), dtype=float)
    for i in range(N):
        cycle[i, (i + 1) % N] = cycle[(i + 1) % N, i] = 1

    regular = np.zeros_like(cycle)
    for i, j in REGULAR_EDGES:
        regular[i, j] = regular[j, i] = 1

    graphs = dict(cycle=cycle, regular=regular, wellmixed=np.ones((N, N)) - np.eye(N))
    for name, graph in graphs.items():
        digest = hashlib.sha256(graph.astype('<f8',copy=False).tobytes()).hexdigest()
        if digest != GRAPH_HASHES[name]:
            raise ValueError(f'原网络发生变化：{name}')
    return graphs

def make_sample(epsilon, size):
    """生成网格突变者，并标出其中 16 个角点的位置。"""
    if (not np.isfinite(epsilon) or not EPSILON_MARGIN <= epsilon <= .5-EPSILON_MARGIN
            or type(size) is not int or size < 2):
        raise ValueError('Require 1e-4 <= epsilon <= 0.4999 and integer grid size >= 2')
    digits = np.array(list(product(range(size), repeat=4)), dtype=int)
    probabilities = np.linspace(epsilon, 1-epsilon, size)[digits]
    corner_ids = (BITS * (size-1)) @ (size**np.arange(3, -1, -1))
    self_cooperation = cooperation_rates(probabilities, probabilities)[0]
    return dict(epsilon=epsilon, size=size, digits=digits, probabilities=probabilities,
                corner_ids=corner_ids, self_c=self_cooperation)

def resident_components(sample, resident):
    """四种相遇：突变者自身、突变者遇居民、居民遇突变者、居民自身。"""
    resident_id = int(sample['corner_ids'][resident])
    resident_strategy = sample['probabilities'][resident_id]
    paired = np.column_stack(cooperation_rates(sample['probabilities'], resident_strategy))
    return dict(mutant_self=sample['self_c'], mutant_against_resident=paired[:, 0],
                resident_against_mutant=paired[:, 1], resident_self=sample['self_c'][resident_id])

def linear_constraints(cooperation, tau, resident_grid_id):
    """入侵条件 F/c = alpha*(b/c) + beta <= 0。"""
    t1, t2, t3 = tau
    mm = cooperation['mutant_self']
    mr = cooperation['mutant_against_resident']
    rm = cooperation['resident_against_mutant']
    rr = cooperation['resident_self']

    # F = t1(s_MR-s_RR) + t2(s_RM-s_RR) + t3(s_MM-s_MR-s_RM+s_RR)
    # mr/rm 表示“自己的合作概率”，收益里的 benefit 因而使用另一方。
    alpha = t1*(rm-rr) + t2*(mr-rr) + t3*(mm-rm-mr+rr)
    beta = -(t1*(mr-rr) + t2*(rm-rr) + t3*(mm-mr-rm+rr))
    np.testing.assert_allclose([alpha[resident_grid_id], beta[resident_grid_id]], 0, atol=1e-10, rtol=0)
    alpha[resident_grid_id] = beta[resident_grid_id] = 0
    return alpha, beta

def intersect_constraints(alpha, beta, ids, tolerance=1e-10):
    """在 b/c>1 上求交集，保留空集、无穷上界和单点区间。"""
    ids = np.asarray(ids, dtype=int)
    a, b = np.asarray(alpha)[ids], np.asarray(beta)[ids]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite invasion coefficients')
    lower_mask, upper_mask = a < -tolerance, a > tolerance
    constant = ~(lower_mask | upper_mask)
    roots = np.full(len(ids), np.nan)
    np.divide(-b, a, out=roots, where=~constant)
    lower = max(1., float(np.max(roots[lower_mask]))) if lower_mask.any() else 1.
    upper = float(np.min(roots[upper_mask])) if upper_mask.any() else np.inf
    fail_ids = ids[constant & (b > tolerance)].tolist()
    feasible = bool(not fail_ids and upper >= lower and upper > 1)
    def binding(value, mask):
        if not np.isfinite(value):
            return []
        return ids[mask & np.isclose(roots, value, atol=1e-8, rtol=1e-9)].tolist()
    return dict(feasible=feasible, lower_bc=lower if feasible else None,
                upper_bc=upper if feasible and np.isfinite(upper) else None,
                upper_unbounded=bool(feasible and not np.isfinite(upper)),
                lower_inclusive=bool(feasible and lower > 1), upper_inclusive=bool(feasible and np.isfinite(upper)),
                singleton=bool(feasible and lower == upper),
                lower_binding_ids=binding(lower, lower_mask), upper_binding_ids=binding(upper, upper_mask),
                constant_failure_ids=fail_ids, lower_envelope=lower,
                upper_envelope=upper if np.isfinite(upper) else None,
                constraints_count=len(ids),
                numerically_neutral_constraints=int(np.count_nonzero(constant & (abs(b) <= tolerance))))

def opponent_record(sample, index):
    digits = sample['digits'][index]
    is_corner = bool(np.all((digits == 0) | (digits == sample['size']-1)))
    corner_id = int(np.dot(digits == sample['size']-1, [8, 4, 2, 1])) if is_corner else None
    return dict(grid_id=int(index), probabilities=sample['probabilities'][index].tolist(),
                corner_id=corner_id, name=strategy_name(corner_id) if is_corner else 'grid_'+str(index))

def evaluate_resident(sample, structures, resident, invaders='both', tolerance=1e-10):
    components = resident_components(sample, resident)
    resident_gid = int(sample['corner_ids'][resident])
    all_ids = np.delete(np.arange(len(sample['probabilities'])), resident_gid)
    corner_ids = np.delete(sample['corner_ids'], resident)
    records, constraints = [], {}
    for network, structure in structures.items():
        alpha, beta = linear_constraints(components, structure['tau'], resident_gid)
        # 极小但非零的斜率可能对应很大的有限边界，不能随意按零处理。
        noise_floor = 256*np.finfo(float).eps*sum(abs(t) for t in structure['tau'])
        ambiguous = ((abs(alpha) > noise_floor) & (abs(alpha) <= tolerance))
        ambiguous |= ((abs(alpha) <= noise_floor) & (abs(beta) > noise_floor)
                      & (abs(beta) <= tolerance))
        if ambiguous.any():
            raise ArithmeticError(f'Unresolved near-zero invasion coefficient: {network}, '
                                  f'epsilon={sample["epsilon"]}, resident={resident}. '
                                  'Use a higher-precision calculation for these parameters.')
        constraints[network] = (alpha, beta)
        for label, ids in [('corners', corner_ids), ('grid', all_ids)]:
            if label == 'grid' and invaders == 'corners':
                continue
            region = intersect_constraints(alpha, beta, ids, tolerance)
            region['lower_binding'] = [opponent_record(sample, i) for i in region['lower_binding_ids']]
            region['upper_binding'] = [opponent_record(sample, i) for i in region['upper_binding_ids']]
            records.append(dict(network=network, epsilon=sample['epsilon'], resident_id=resident,
                                resident_bits=bit_label(resident), resident_name=strategy_name(resident),
                                invader_set=label, region=region))
    return records, constraints


def interval_text(region, digits=10):
    """Format the intersection with the open domain b/c>1."""
    if not region['feasible']:
        return 'empty'
    left='[' if region['lower_inclusive'] else '('
    upper=region['upper_bc']
    return f"{left}{region['lower_bc']:.{digits}g}, "+('inf)' if upper is None else f'{upper:.{digits}g}]')


def run_analysis(args):
    """Compute network coefficients and all requested finite-grid intervals."""
    started=time.perf_counter()
    graphs=make_graphs()
    structures={}
    for name in args.networks:
        print(f'Calculating finite-network coalescence coefficients: {name}',flush=True)
        tau=network_coefficients(graphs[name])
        structures[name]=dict(tau=list(tau),N=N,degree=int(graphs[name][0].sum()),
            adjacency_sha256=GRAPH_HASHES[name],method='Full two- and three-lineage coalescence equations',
            solver_rtol=1e-12,solver_atol=1e-12,maximum_allowed_absolute_residual=1e-8)
        print(f'  tau = {tau}',flush=True)
    resident_ids=[int(bits,2) for bits in args.residents]
    records,arrays=[],{}
    size=args.grid_size if args.invaders=='both' else 2
    for index,epsilon in enumerate(args.epsilon):
        sample=make_sample(epsilon,size)
        arrays[f'e{index}_probabilities']=sample['probabilities']
        arrays[f'e{index}_corner_ids']=sample['corner_ids']
        for resident in resident_ids:
            found,constraints=evaluate_resident(sample,structures,resident,args.invaders,args.zero_tolerance)
            records.extend(found)
            for network,(alpha,beta) in constraints.items():
                key=f'e{index}_{network}_resident{resident:02d}'
                arrays[key+'_alpha']=alpha
                arrays[key+'_beta']=beta
        print(f'Finished epsilon={epsilon:g}: {len(resident_ids)} residents, {size**4} grid strategies',flush=True)
    metadata=dict(N=N,cost=1,epsilon=args.epsilon,networks=args.networks,resident_ids=resident_ids,
        grid_size=size,invader_sets=args.invaders,zero_tolerance=args.zero_tolerance,
        strategy_order=['CC','CD','DC','DD'],effective_probabilities=True,
        formula='F/c = alpha*(b/c)+beta <= 0',domain='b/c > 1',structures=structures,
        scope='First-order weak-selection invasion robustness against the specified finite mutant sets. '
              'Identical residents are excluded (15 or grid_size**4-1 distinct mutants). '
              'Finite equality endpoints are first-order neutral. This is not recurrent-mutation '
              'abundance, a strict-ESS test, or a proof over a continuous strategy space.',
        source_sha256=SOURCE_SHA256,results=records,elapsed_seconds=time.perf_counter()-started)
    output=args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    with (output/'robustness_intervals.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['network','epsilon','resident_id','resident_bits','resident_name','invader_set',
                         'distinct_invaders','feasible','lower_bc','upper_bc','upper_unbounded',
                         'lower_inclusive','upper_inclusive','singleton','interval'])
        for row in records:
            region=row['region']
            writer.writerow([row[k] for k in ('network','epsilon','resident_id','resident_bits','resident_name','invader_set')]
                +[region[k] for k in ('constraints_count','feasible','lower_bc','upper_bc','upper_unbounded',
                                     'lower_inclusive','upper_inclusive','singleton')]+[interval_text(region)])
    text=json.dumps(metadata,indent=2,allow_nan=False)+'\n'
    (output/'parameters_and_intervals.json').write_text(text,encoding='utf-8')
    arrays['metadata_json']=np.array(json.dumps(metadata,allow_nan=False))
    np.savez_compressed(output/'invasion_constraints.npz',**arrays)
    print(f'Wrote {len(records)} intervals to {output} in {time.perf_counter()-started:.2f} s',flush=True)
    return metadata


def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--networks',nargs='+',choices=list(NETWORK_NAMES),default=list(NETWORK_NAMES))
    parser.add_argument('--epsilon',nargs='+',type=float,default=[.1,.3])
    parser.add_argument('--residents',nargs='+',choices=[format(i,'04b') for i in range(16)],
                        default=[format(i,'04b') for i in range(16)],metavar='BITS',
                        help='One or more resident bit strings, e.g. 1111 (ALLC), 0000 (ALLD), 1001 (WSLS), 1000 (Grim); default all 16')
    parser.add_argument('--invaders',choices=['both','corners'],default='both',
                        help='both: corners and Cartesian grid; corners: only the other 15 corners')
    parser.add_argument('--grid-size',type=int,default=9,help='Values per coordinate, including endpoints (default 9)')
    parser.add_argument('--zero-tolerance',type=float,default=1e-10)
    parser.add_argument('--output-dir',type=Path,default=Path('results/theory'))
    args=parser.parse_args(argv)
    if any(not np.isfinite(e) or not EPSILON_MARGIN<=e<=.5-EPSILON_MARGIN for e in args.epsilon):
        parser.error('Float64 calculation requires 1e-4 <= epsilon <= 0.4999')
    if args.grid_size<2 or not np.isfinite(args.zero_tolerance) or args.zero_tolerance<=0:
        parser.error('Require grid-size >= 2 and finite positive zero-tolerance')
    if any(len(set(v))!=len(v) for v in [args.networks,args.epsilon,args.residents]):
        parser.error('Do not repeat networks, epsilon values or residents')
    return args


if __name__=='__main__':
    run_analysis(arguments())
