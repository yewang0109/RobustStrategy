#!/usr/bin/env python3
"""Sixteen memory-one corner strategies competing simultaneously on fixed N=50 graphs.

Production model functions are preserved verbatim. Use the package local runner.
Public API: make_graphs, Engine, condition_seed, run_trajectory,
run_checkpointed_trajectory and atomic_npz. No plotting or inference.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import njit

DEFAULTS = {
    'networks': ['cycle', 'regular', 'wellmixed'],
    'epsilon': [0.1, 0.3],
    'mu': [0.005, 0.01, 0.1],
    'bc': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
           6.0, 7.0, 8.0, 9.0, 10.0,
           12.0, 14.0, 16.0, 18.0, 20.0,
           25.0, 30.0, 35.0, 40.0],
    'cost': 1.0,
    'delta': 0.01,
    'replicates': 1000,
    'steps': 2_000_000_000,    # 每条轨迹预热后的采样更新数；不是 sweeps
    'burn_in': 1_000_000,      # 每条轨迹预热100万次节点更新
    'blocks': 128,            # 每块15,625,000次；分块用于诊断，不增加独立样本
    'seed': 20261006,
    'chunk_steps': 1_000_000,
    'checkpoint_seconds': 30.0,
}


N = 50


K = 16


STRATEGY_MODEL = 'memory-one corners'


STRATEGY_COORDINATES = ('CC', 'CD', 'DC', 'DD')


NETWORK_NAMES = {'cycle': 'Cycle (k=2)', 'regular': 'Regular (k=4)',
                 'wellmixed': 'Well-mixed (k=49)'}


REGULAR_EDGES = (
    (0, 1), (0, 36), (0, 39), (0, 43), (1, 2), (1, 33),
    (1, 34), (2, 3), (2, 10), (2, 12), (3, 9), (3, 35),
    (3, 39), (4, 5), (4, 6), (4, 9), (4, 19), (5, 16),
    (5, 19), (5, 41), (6, 7), (6, 20), (6, 26), (7, 17),
    (7, 28), (7, 41), (8, 9), (8, 22), (8, 30), (8, 45),
    (9, 45), (10, 11), (10, 25), (10, 37), (11, 29), (11, 32),
    (11, 44), (12, 15), (12, 43), (12, 48), (13, 14), (13, 23),
    (13, 38), (13, 49), (14, 16), (14, 23), (14, 25), (15, 18),
    (15, 46), (15, 47), (16, 35), (16, 40), (17, 43), (17, 46),
    (17, 47), (18, 21), (18, 27), (18, 30), (19, 20), (19, 22),
    (20, 38), (20, 49), (21, 22), (21, 32), (21, 45), (22, 46),
    (23, 24), (23, 25), (24, 25), (24, 36), (24, 39), (26, 30),
    (26, 35), (26, 41), (27, 28), (27, 42), (27, 48), (28, 29),
    (28, 37), (29, 36), (29, 42), (30, 47), (31, 32), (31, 33),
    (31, 34), (31, 38), (32, 40), (33, 34), (33, 48), (34, 44),
    (35, 37), (36, 40), (37, 41), (38, 39), (40, 45), (42, 44),
    (42, 48), (43, 46), (44, 49), (47, 49),
)


def strategy_bits(index):
    """Binary corner ID in focal-player CC, CD, DC, DD order."""
    if not isinstance(index, (int, np.integer)) or not 0 <= index < K:
        raise ValueError('Strategy ID must be an integer from 0 to 15')
    return f'{index:04b}'


def make_strategies(epsilon):
    if not np.isfinite(epsilon) or not 0 < epsilon < .5:
        raise ValueError('Require 0 < epsilon < 0.5')
    bits = ((np.arange(K)[:, None] >> np.arange(3, -1, -1)) & 1)
    return np.ascontiguousarray(epsilon + (1-2*epsilon)*bits, dtype=np.float64)


def stationary_distribution(p, q):
    """Unique stationary law in focal CC, CD, DC, DD order.

    q is supplied in the opponent's own CC, CD, DC, DD coordinates, so its
    CD and DC entries must be interchanged in the focal player's states.
    Interior cooperation probabilities ensure an irreducible aperiodic chain.
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if (p.shape != (4,) or q.shape != (4,)
            or not np.all(np.isfinite(p)) or not np.all(np.isfinite(q))
            or np.any(p <= 0) or np.any(p >= 1) or np.any(q <= 0) or np.any(q >= 1)):
        raise ValueError('Two memory-one four-vectors with probabilities strictly in (0,1) are required')
    opponent = q[[0, 2, 1, 3]]
    transition = np.column_stack((p*opponent, p*(1-opponent),
                                  (1-p)*opponent, (1-p)*(1-opponent)))
    equations = transition.T - np.eye(4)
    equations[-1] = 1.
    stationary = np.linalg.solve(equations, np.array([0., 0., 0., 1.]))
    if (not np.all(np.isfinite(stationary)) or np.min(stationary) < -1e-12
            or not np.allclose(stationary @ transition, stationary, atol=1e-12, rtol=1e-12)):
        raise ArithmeticError('Stationary distribution solve failed')
    return stationary


def make_payoffs(strategies, b, c):
    strategies = np.asarray(strategies, dtype=float)
    if strategies.ndim != 2 or strategies.shape[1] != 4:
        raise ValueError('Memory-one strategies must have shape [K,4]')
    if not np.isfinite(b) or not np.isfinite(c):
        raise ValueError('Benefit and cost must be finite')
    reward = np.array([b-c, -c, b, 0.])
    game = np.empty((len(strategies), len(strategies)), dtype=np.float64)
    for i, p in enumerate(strategies):
        for j, q in enumerate(strategies):
            game[i, j] = stationary_distribution(p, q) @ reward
    return np.ascontiguousarray(game)


def new_rng(seed, condition, replicate, phase=0):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [int(seed), int(condition), int(replicate), int(phase)])))


def initial_state(n, k, rng):
    labels = rng.permutation(k)
    state = np.asarray(labels[np.arange(n) % k], dtype=np.int64)
    rng.shuffle(state)
    return state, np.bincount(state, minlength=k).astype(np.int64)


@njit(cache=True)
def recompute_totals(game, counts):
    k = len(counts)
    totals = np.zeros(k, np.float64)
    for i in range(k):
        for j in range(k):
            totals[i] += game[i, j]*counts[j]
        totals[i] -= game[i, i]
    return totals


@njit(cache=True)
def flush(counts, last, sums, block, end):
    for s in range(len(counts)):
        sums[block, s] += counts[s]*(end-last[s])
        last[s] = end


@njit(cache=True)
def graph_chunk(neighbors, game, delta, mu, state, counts, rng,
                done, burn, length, sums, steps, richness_sums):
    n, degree = neighbors.shape
    k = len(counts)
    last = np.full(k, max(0, done-burn), np.int64)
    richness = np.count_nonzero(counts)
    richness_last = max(0, done-burn)
    weights = np.empty(degree, np.float64)
    for offset in range(steps):
        i = rng.integers(0, n)
        old = state[i]
        if rng.random() < mu:
            new = rng.integers(0, k)
        else:
            total = 0.
            for position in range(degree):
                parent = neighbors[i, position]
                pi = 0.
                for neighbor in neighbors[parent]:
                    pi += game[state[parent], state[neighbor]]
                weights[position] = 1+delta*pi/degree
                total += weights[position]
            draw = rng.random()*total
            cumulative = 0.
            new = state[neighbors[i, degree-1]]
            for position in range(degree):
                cumulative += weights[position]
                if cumulative > draw:
                    new = state[neighbors[i, position]]
                    break
        q = done+offset-burn
        if new != old:
            richness_change = (1 if counts[new] == 0 else 0) - (1 if counts[old] == 1 else 0)
            if richness_change != 0:
                if q >= 0:
                    richness_sums[q//length] += richness*(q-richness_last)
                    richness_last = q
                richness += richness_change
            if q >= 0:
                block = q//length
                sums[block, old] += counts[old]*(q-last[old])
                sums[block, new] += counts[new]*(q-last[new])
                last[old] = q; last[new] = q
            state[i] = new
            counts[old] -= 1; counts[new] += 1
        if q >= 0 and (q+1) % length == 0:
            flush(counts, last, sums, q//length, q+1)
            richness_sums[q//length] += richness*(q+1-richness_last)
            richness_last = q+1
    end = done+steps-burn
    if end > 0 and end % length:
        flush(counts, last, sums, (end-1)//length, end)
        richness_sums[(end-1)//length] += richness*(end-richness_last)


@njit(cache=True)
def mixed_chunk(game, delta, mu, state, counts, totals, rng,
                done, burn, length, sums, steps, refresh, richness_sums):
    n, k = len(state), len(counts)
    last = np.full(k, max(0, done-burn), np.int64)
    richness = np.count_nonzero(counts)
    richness_last = max(0, done-burn)
    weights = np.empty(k, np.float64)
    for offset in range(steps):
        i = rng.integers(0, n)
        old = state[i]
        if rng.random() < mu:
            new = rng.integers(0, k)
        else:
            total = 0.
            for s in range(k):
                candidates = counts[s]-(1 if s == old else 0)
                weights[s] = candidates*(1+delta*totals[s]/(n-1))
                total += weights[s]
            draw = rng.random()*total
            cumulative = 0.
            new = old
            for s in range(k):
                if weights[s] > 0:
                    new = s  # valid fallback if cumulative rounding reaches the endpoint
                cumulative += weights[s]
                if cumulative > draw:
                    new = s
                    break
        q = done+offset-burn
        if new != old:
            richness_change = (1 if counts[new] == 0 else 0) - (1 if counts[old] == 1 else 0)
            if richness_change != 0:
                if q >= 0:
                    richness_sums[q//length] += richness*(q-richness_last)
                    richness_last = q
                richness += richness_change
            if q >= 0:
                block = q//length
                sums[block, old] += counts[old]*(q-last[old])
                sums[block, new] += counts[new]*(q-last[new])
                last[old] = q; last[new] = q
            state[i] = new
            counts[old] -= 1; counts[new] += 1
            for s in range(k):
                totals[s] += game[s, new]-game[s, old]
        if (done+offset+1) % refresh == 0:
            fresh = recompute_totals(game, counts)
            for s in range(k):
                totals[s] = fresh[s]
        if q >= 0 and (q+1) % length == 0:
            flush(counts, last, sums, q//length, q+1)
            richness_sums[q//length] += richness*(q+1-richness_last)
            richness_last = q+1
    end = done+steps-burn
    if end > 0 and end % length:
        flush(counts, last, sums, (end-1)//length, end)
        richness_sums[(end-1)//length] += richness*(end-richness_last)


class Engine:
    def __init__(self, W, condition, settings):
        self.W, self.condition, self.settings = W, condition, settings
        self.n = len(W)
        self.strategies = make_strategies(condition['epsilon'])
        self.k = len(self.strategies)
        self.game = make_payoffs(self.strategies, condition['bc']*settings['cost'], settings['cost'])
        self.delta, self.mu = condition['delta'], condition['mu']
        if np.min(1+self.delta*self.game) <= 0:
            raise ValueError('Linear fitness must be positive for every possible average payoff')
        self.neighbors = np.array([np.flatnonzero(row) for row in W], dtype=np.int64)
        self.is_count = condition['network'] == 'wellmixed'
        if self.is_count and not np.array_equal(W, np.ones_like(W)-np.eye(self.n)):
            raise ValueError('Well-mixed acceleration requires the complete graph')

    def initialize(self, rng):
        state, counts = initial_state(self.n, self.k, rng)
        totals = recompute_totals(self.game, counts) if self.is_count else np.zeros(self.k, np.float64)
        return state, counts, totals

    def advance(self, state, counts, totals, rng, done, burn, length, sums, steps,
                richness_sums=None):
        if richness_sums is None:
            richness_sums = np.zeros(sums.shape[0], dtype=np.int64)
        elif richness_sums.shape != (sums.shape[0],) or richness_sums.dtype != np.dtype('int64'):
            raise ValueError('richness_sums must be an int64 array with one entry per time block')
        if self.is_count:
            mixed_chunk(self.game, self.delta, self.mu, state, counts, totals, rng,
                        done, burn, length, sums, steps, self.settings['payoff_refresh_steps'],
                        richness_sums)
        else:
            graph_chunk(self.neighbors, self.game, self.delta, self.mu, state,
                        counts, rng, done, burn, length, sums, steps, richness_sums)

    def compile(self):
        rng = new_rng(0, 0, 0, 99)
        state, counts, totals = self.initialize(rng)
        self.advance(state, counts, totals, rng, 0, 0, 10, np.zeros((1, self.k), np.int64), 1)


def make_graphs():
    """Return the same fixed three N=50 graphs used in the earlier simulations."""
    if N != 50:
        raise ValueError('These fixed graph definitions require N=50.')
    cycle = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        cycle[i, (i+1) % N] = cycle[(i+1) % N, i] = 1
    regular = np.zeros_like(cycle)
    for i, j in REGULAR_EDGES:
        regular[i, j] = regular[j, i] = 1
    mixed = np.ones_like(cycle) - np.eye(N, dtype=np.int64)
    graphs = dict(cycle=cycle, regular=regular, wellmixed=mixed)
    for name, graph in graphs.items():
        degree = {'cycle': 2, 'regular': 4, 'wellmixed': N-1}[name]
        if not (np.array_equal(graph, graph.T) and not np.any(np.diag(graph))
                and np.all(graph.sum(axis=1) == degree)):
            raise ValueError(f'Invalid graph: {name}')
    return graphs


def condition_seed(base_seed, condition, replicate):
    """Parameter-based streams do not change when sweep order/subsets change."""
    encoded = json.dumps(condition, sort_keys=True, separators=(',', ':')).encode()
    words = np.frombuffer(hashlib.sha256(encoded).digest(), dtype='<u4').tolist()
    seed = np.random.SeedSequence([int(base_seed), int(replicate)] + words)
    return np.random.Generator(np.random.PCG64(seed))


def run_trajectory(engine, rng, steps, burn_in, blocks):
    state, counts, totals = engine.initialize(rng)
    sums = np.zeros((blocks, K), dtype=np.int64)
    richness_sums = np.zeros(blocks, dtype=np.int64)
    engine.advance(state, counts, totals, rng, 0, burn_in, steps//blocks,
                   sums, burn_in+steps, richness_sums)
    if np.any(sums < 0) or not np.all(sums.sum(axis=1) == N*(steps//blocks)):
        raise RuntimeError('Sample accounting failed.')
    if not np.array_equal(counts, np.bincount(state, minlength=K)):
        raise RuntimeError('Population accounting failed.')
    return sums, counts, richness_sums


def run_checkpointed_trajectory(engine, rng, steps, burn_in, blocks, path,
                                signature, chunk_steps, checkpoint_seconds):
    """Resume exactly, or start a trajectory and atomically save safe boundaries.

    The signature must identify the model, graph, condition, seed and replicate.
    An interrupted kernel may have changed arrays without reporting how many
    updates finished. In that case leave the previous durable checkpoint intact;
    never save those partially changed arrays under an incorrect ``done`` value.
    """
    path = Path(path)
    n, k = engine.n, engine.k
    if (steps <= 0 or burn_in < 0 or blocks < 2 or steps % blocks
            or chunk_steps <= 0 or not np.isfinite(checkpoint_seconds)
            or checkpoint_seconds < 0):
        raise ValueError('Invalid trajectory/checkpoint settings')
    if n*steps > np.iinfo(np.int64).max or steps+burn_in > np.iinfo(np.int64).max:
        raise ValueError('Trajectory counters exceed int64 capacity')
    if not isinstance(rng.bit_generator, np.random.PCG64):
        raise ValueError('Checkpoint restoration requires PCG64')
    signature_json = json.dumps(signature, sort_keys=True, separators=(',', ':'),
                                ensure_ascii=False, allow_nan=False)
    expected_parameters = np.array([n, k, steps, burn_in, blocks], dtype=np.int64)
    length, target = steps//blocks, steps+burn_in

    def validate(done, state, counts, totals, sums, richness_sums):
        if not 0 <= done <= target:
            raise ValueError('Checkpoint update counter is out of range')
        for name, value, shape in [('state', state, (n,)), ('counts', counts, (k,)),
                                   ('sums', sums, (blocks, k)),
                                   ('richness_sums', richness_sums, (blocks,))]:
            if value.shape != shape or value.dtype != np.dtype('int64'):
                raise ValueError(f'Checkpoint {name} has the wrong shape or dtype')
        if totals.shape != (k,) or totals.dtype != np.dtype('float64') or not np.isfinite(totals).all():
            raise ValueError('Checkpoint payoff totals are invalid')
        if np.any(state < 0) or np.any(state >= k):
            raise ValueError('Checkpoint strategy labels are invalid')
        if not np.array_equal(counts, np.bincount(state, minlength=k)):
            raise ValueError('Checkpoint population counts disagree with its state')
        sampled = max(0, done-burn_in)
        expected = n*np.clip(sampled-np.arange(blocks, dtype=np.int64)*length, 0, length)
        if np.any(sums < 0) or not np.array_equal(sums.sum(axis=1), expected):
            raise ValueError('Checkpoint sampled counts disagree with its update counter')
        elapsed_per_block = expected//n
        if (np.any(richness_sums < elapsed_per_block)
                or np.any(richness_sums > min(n, k)*elapsed_per_block)):
            raise ValueError('Checkpoint coexistence sums disagree with its sampled duration')

    if path.exists():
        # No pickles, and no silent restart when a checkpoint does not match.
        try:
            with np.load(path, allow_pickle=False) as saved:
                if saved['schema'].shape != () or int(saved['schema']) != 2:
                    raise ValueError('Checkpoint schema 2 with coexistence statistics is required')
                if saved['signature_json'].shape != () or str(saved['signature_json']) != signature_json:
                    raise ValueError('Checkpoint signature does not match this trajectory')
                if not np.array_equal(saved['parameters'], expected_parameters):
                    raise ValueError('Checkpoint sampling parameters do not match')
                if saved['done'].shape != () or saved['done'].dtype != np.dtype('int64'):
                    raise ValueError('Checkpoint update counter is invalid')
                done = int(saved['done'])
                state, counts, totals, sums = (saved[key].copy()
                                               for key in ('state', 'counts', 'totals', 'sums'))
                richness_sums = saved['richness_sums'].copy()
                rng_state = json.loads(str(saved['rng_state_json']))
            validate(done, state, counts, totals, sums, richness_sums)
            checker = np.random.PCG64()
            checker.state = rng_state
            rng.bit_generator.state = rng_state
        except Exception as exc:
            raise ValueError(f'Cannot resume checkpoint {path}: {exc}') from exc
        if done == target:
            return sums, counts, richness_sums
    else:
        done = 0
        state, counts, totals = engine.initialize(rng)
        sums = np.zeros((blocks, k), dtype=np.int64)
        richness_sums = np.zeros(blocks, dtype=np.int64)
        validate(done, state, counts, totals, sums, richness_sums)

    def save():
        validate(done, state, counts, totals, sums, richness_sums)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+'.tmp')
        # Passing a file handle prevents NumPy appending an extra .npz suffix.
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, schema=np.array(2, dtype=np.int64),
                                signature_json=np.array(signature_json), parameters=expected_parameters,
                                done=np.array(done, dtype=np.int64), state=state, counts=counts,
                                totals=totals, sums=sums, richness_sums=richness_sums,
                                rng_state_json=np.array(json.dumps(rng.bit_generator.state)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    # Even interruption during the first kernel call has a valid restart point.
    if not path.exists():
        save()
    last_checkpoint = perf_counter()
    at_safe_boundary = True
    try:
        while done < target:
            size = min(chunk_steps, target-done)
            at_safe_boundary = False
            engine.advance(state, counts, totals, rng, done, burn_in, length, sums, size, richness_sums)
            done += size
            at_safe_boundary = True
            if done == target or perf_counter()-last_checkpoint >= checkpoint_seconds:
                save()
                last_checkpoint = perf_counter()
                if done < target:
                    print(f'    {path.stem}: {done:,}/{target:,} updates ({done/target:.1%}); checkpoint saved', flush=True)
    except BaseException:
        if at_safe_boundary:
            save()
        # Otherwise resume from the last safe on-disk checkpoint on the next run.
        raise
    return sums, counts, richness_sums


def strategy_name(index):
    return {0: 'ALLDe', 5: 'ATFTe', 8: 'Grim-e', 9: 'WSLS-e', 10: 'TFTe', 15: 'ALLCe'}.get(index, f'S{index:02d}')


def atomic_npz(path, **arrays):
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f, **arrays)
    temporary.replace(path)
