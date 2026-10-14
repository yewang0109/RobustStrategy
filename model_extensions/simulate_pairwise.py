#!/usr/bin/env python3
"""ALLC versus one of 35 reactive opponents on the fixed N=50 cycle; local columns are [opponent, ALLC].

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
import tempfile

import numpy as np
from numba import njit

N = 50


GRID_SIZE = 6


K = 2


ALLC_ID = GRID_SIZE ** 2 - 1


OPPONENT_IDS = tuple(range(ALLC_ID))


NETWORK_NAMES = {'cycle': 'Cycle (k=2)'}


MODEL_VERSION = 'reactive36-pairwise-allc-cycle-db-v1'


CONFIG_KEYS = ['networks', 'epsilon', 'mu', 'bc', 'cost', 'delta',
               'replicates', 'steps', 'burn_in', 'blocks', 'seed']


DEFAULTS = {
    'networks': ['cycle'], 'epsilon': [0.1], 'mu': [0.01],
    'bc': [3.0, 5.0, 8.0], 'cost': 1.0, 'delta': 0.01,
    'replicates': 1000, 'steps': 2_000_000_000, 'burn_in': 1_000_000,
    'blocks': 128, 'seed': 20261015,
}


def make_strategies(epsilon, grid_size=GRID_SIZE):
    if not 0 < epsilon < .5 or grid_size < 2:
        raise ValueError('Require 0 < epsilon < 0.5 and at least two grid coordinates')
    axis = np.linspace(epsilon, 1-epsilon, grid_size)
    return np.array([(p, q) for p in axis for q in axis], dtype=np.float64)


def make_payoffs(strategies, b, c):
    strategies = np.asarray(strategies, dtype=np.float64)
    if (strategies.ndim != 2 or strategies.shape[1] != 2
            or not np.isfinite(strategies).all()
            or np.any(strategies <= 0) or np.any(strategies >= 1)
            or not np.isfinite(b) or not np.isfinite(c)):
        raise ValueError('Payoffs require interior reactive probabilities and finite b,c')
    p, q = strategies.T
    r = p-q
    cooperation = (q[:, None]+r[:, None]*q[None, :])/(1-r[:, None]*r[None, :])
    return np.ascontiguousarray(b*cooperation.T-c*cooperation)


def new_rng(seed, condition, replicate, phase=0):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [int(seed), int(condition), int(replicate), int(phase)])))


def condition_seed(base_seed, condition, replicate):
    """The condition includes opponent_id, so different opponents never share streams."""
    if 'opponent_id' not in condition:
        raise ValueError('A pairwise condition must identify its opponent')
    encoded = json.dumps(condition, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode()
    words = np.frombuffer(hashlib.sha256(encoded).digest(), dtype='<u4').tolist()
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [int(base_seed), int(replicate)] + words)))


def initial_state(n, k, rng):
    if k != K or n % 2:
        raise ValueError('Pairwise trajectories require two types and an even population')
    state = np.repeat(np.arange(k, dtype=np.int64), n//k)
    rng.shuffle(state)
    return state, np.bincount(state, minlength=k).astype(np.int64)


def make_graphs():
    """Same undirected, unweighted 50-node cycle as the previous pairwise code."""
    cycle = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        cycle[i, (i+1) % N] = cycle[(i+1) % N, i] = 1
    return {'cycle': cycle}


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


class Engine:
    def __init__(self, W, condition, settings):
        self.W = np.asarray(W)
        self.condition, self.settings = condition, settings
        self.n, self.k = len(W), K
        if condition['network'] != 'cycle':
            raise ValueError('Figure R9 uses only the cycle')
        if (self.W.shape != (self.n, self.n) or self.n < 4 or self.n % 2
                or not np.array_equal(self.W, self.W.T) or np.any(np.diag(self.W))
                or not np.all((self.W == 0) | (self.W == 1))
                or not np.all(self.W.sum(axis=1) == 2)):
            raise ValueError('Require an undirected simple even cycle with degree two')
        # Degree two alone could also describe several disjoint cycles.
        visited, stack = set(), [0]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(np.flatnonzero(self.W[node]).tolist())
        if len(visited) != self.n:
            raise ValueError('Cycle must be connected')
        if settings['grid_size'] != GRID_SIZE:
            raise ValueError('Figure R9 requires the 6 by 6 reactive grid')
        opponent_id = condition['opponent_id']
        if type(opponent_id) is not int or opponent_id not in OPPONENT_IDS:
            raise ValueError('opponent_id must be 0..34; ALLC versus itself is excluded')
        self.opponent_id = opponent_id
        self.global_ids = np.array([opponent_id, ALLC_ID], dtype=np.int64)
        self.strategies = make_strategies(condition['epsilon'], GRID_SIZE)[self.global_ids]
        self.game = make_payoffs(self.strategies,
                                 condition['bc']*settings['cost'], settings['cost'])
        self.delta, self.mu = condition['delta'], condition['mu']
        if (not np.isfinite(self.delta) or self.delta < 0
                or not np.isfinite(self.mu) or not 0 <= self.mu <= 1
                or np.min(1+self.delta*self.game) <= 0):
            raise ValueError('Require valid mutation rate and positive linear fitness')
        self.neighbors = np.array([np.flatnonzero(row) for row in self.W], dtype=np.int64)
        self.is_count = False

    def initialize(self, rng):
        state, counts = initial_state(self.n, self.k, rng)
        # Kept for worker/checkpoint API compatibility; graph payoffs are computed
        # from the actual pre-update state, avoiding incremental payoff drift.
        return state, counts, np.zeros(self.k, np.float64)

    def advance(self, state, counts, totals, rng, done, burn, length, sums, steps,
                richness_sums=None):
        if (done < 0 or burn < 0 or length <= 0 or steps < 0
                or done+steps > burn+length*sums.shape[0]
                or sums.ndim != 2 or sums.shape[1] != self.k
                or sums.dtype != np.dtype('int64')):
            raise ValueError('Invalid update range or int64 block accumulator')
        if richness_sums is None:
            richness_sums = np.zeros(sums.shape[0], dtype=np.int64)
        elif richness_sums.shape != (sums.shape[0],) or richness_sums.dtype != np.dtype('int64'):
            raise ValueError('richness_sums must be an int64 array with one entry per time block')
        graph_chunk(self.neighbors, self.game, self.delta, self.mu, state,
                    counts, rng, done, burn, length, sums, steps, richness_sums)

    def compile(self):
        rng = new_rng(0, 0, 0, 99)
        state, counts, totals = self.initialize(rng)
        self.advance(state, counts, totals, rng, 0, 0, 10,
                     np.zeros((1, self.k), np.int64), 1)


def run_trajectory(engine, rng, steps, burn_in, blocks):
    if steps <= 0 or burn_in < 0 or blocks <= 0 or steps % blocks:
        raise ValueError('Require positive sampled length divisible by block count')
    state, counts, totals = engine.initialize(rng)
    sums = np.zeros((blocks, engine.k), dtype=np.int64)
    richness = np.zeros(blocks, dtype=np.int64)
    engine.advance(state, counts, totals, rng, 0, burn_in, steps//blocks,
                   sums, burn_in+steps, richness)
    if not np.all(sums.sum(axis=1) == engine.n*(steps//blocks)):
        raise RuntimeError('Sample accounting failed')
    if not np.array_equal(counts, np.bincount(state, minlength=engine.k)):
        raise RuntimeError('Population accounting failed')
    return sums, counts, richness


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


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
