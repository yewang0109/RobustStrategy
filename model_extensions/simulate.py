#!/usr/bin/env python3
"""Portable local entry point for the three published competition models.

Default: a short demonstration, not the sampling effort used in the paper.
Use --profile paper for the recorded parameter grid and independent seed batches.
No Slurm or external project files are required. See README.md for examples.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import numba

HERE = Path(__file__).resolve().parent
MODULES = {"reactive": "simulate_reactive36", "memory": "simulate_memory_one16",
           "pairwise": "simulate_pairwise"}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


@contextmanager
def output_lock(folder):
    """An OS lock is released even when an interrupted process exits."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".run.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=MODULES)
    parser.add_argument("--profile", choices=["demo", "paper"], default="demo")
    parser.add_argument("--config", type=Path, default=HERE / "paper_config.json")
    parser.add_argument("--network", choices=["cycle", "regular", "wellmixed"])
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--mu", type=float)
    parser.add_argument("--bc", type=float, help="Benefit-to-cost ratio")
    parser.add_argument("--opponent", type=int, help="Pairwise opponent ID, 0..34")
    parser.add_argument("--batch", help="Select one named paper seed batch")
    parser.add_argument("--replicates", type=int, help="Override repeats per selected batch")
    parser.add_argument("--replicate-start", type=int, default=0,
                        help="First replicate index (default 0); seed streams retain this index")
    parser.add_argument("--steps", type=int, help="Override sampled node updates per trajectory")
    parser.add_argument("--burn-in", type=int, help="Override burn-in node updates per trajectory")
    parser.add_argument("--blocks", type=int, help="Override number of equal recording blocks")
    parser.add_argument("--seed", type=int, help="Custom base seed, replacing the profile's batches")
    parser.add_argument("--chunk-steps", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, help="Output directory; reuse the same command to resume")
    parser.add_argument("--list", action="store_true", help="Print the run plan without running")
    return parser, parser.parse_args()


def make_plan(args, config):
    spec = config["models"][args.model]
    if config["population_size"] != 50:
        raise ValueError("The fixed paper graphs and kernels require population_size=50")
    refresh = config["payoff_refresh_steps"]
    if type(refresh) is not int or refresh < 1:
        raise ValueError("payoff_refresh_steps must be a positive integer")
    if (not np.isfinite(config["cost"]) or config["cost"] <= 0
            or not np.isfinite(config["delta"]) or config["delta"] < 0
            or config["delta"] * config["cost"] >= 1):
        raise ValueError("Require positive cost and 0 <= delta*cost < 1")
    if args.model == "pairwise" and args.network not in (None, "cycle"):
        raise ValueError("The pairwise model implements the paper's cycle design only")
    if args.profile == "paper":
        networks, errors, mutations, ratios = (spec[key] for key in
                                               ("networks", "epsilon", "mu", "bc"))
        batches = [dict(batch) for batch in spec["batches"]]
        steps, burn, blocks = (config[key] for key in ("steps", "burn_in", "blocks"))
        opponents = list(range(35))
    else:
        networks, errors, mutations = ["cycle"], [0.1 if args.model == "pairwise" else 0.3], [0.01]
        ratios = [3.0 if args.model == "pairwise" else 6.0]
        batches = [{"name": "demo", "seed": 20261001, "replicates": 2}]
        steps, burn, blocks, opponents = 100_000, 1_000, 4, [0]
    if args.profile == "paper":
        for selected, available, label in [(args.network, networks, "network"),
                                           (args.epsilon, errors, "epsilon"),
                                           (args.mu, mutations, "mu")]:
            if selected is not None and selected not in available:
                raise ValueError(f"{label} is not in the paper design; use --profile demo for custom parameters")
        allowed_bc = set(ratios) | set(spec.get("regular_extra_bc", []))
        if args.bc is not None and args.bc not in allowed_bc:
            raise ValueError("b/c is not in the paper design; use --profile demo for custom parameters")
    if args.network is not None:
        networks = [args.network]
    if args.epsilon is not None:
        errors = [args.epsilon]
    if args.mu is not None:
        mutations = [args.mu]
    if args.bc is not None:
        ratios = [args.bc]
    if args.opponent is not None:
        if args.model != "pairwise" or not 0 <= args.opponent < 35:
            raise ValueError("--opponent is only valid for pairwise, with IDs 0..34")
        opponents = [args.opponent]
    if args.batch:
        batches = [batch for batch in batches if batch["name"] == args.batch]
        if not batches:
            raise ValueError("Unknown batch name for the chosen model/profile")
    if args.seed is not None:
        if args.batch:
            raise ValueError("Choose --seed or --batch, not both")
        batches = [{"name": "custom", "seed": args.seed,
                    "replicates": sum(batch["replicates"] for batch in batches)}]
    if args.replicates is not None:
        for batch in batches:
            batch["replicates"] = args.replicates
    steps = args.steps if args.steps is not None else steps
    burn = args.burn_in if args.burn_in is not None else burn
    blocks = args.blocks if args.blocks is not None else blocks
    if (steps <= 0 or burn < 0 or blocks < 2 or steps % blocks or args.replicate_start < 0
            or args.chunk_steps <= 0 or not np.isfinite(args.checkpoint_seconds)
            or args.checkpoint_seconds < 0
            or any(batch["replicates"] <= 0 or batch["seed"] < 0 for batch in batches)):
        raise ValueError("Invalid counts/seeds: require steps>0, burn>=0, blocks>=2 dividing steps, repeats>0")
    if steps * 50 >= 2**63 or steps + burn >= 2**63:
        raise ValueError("Update counts exceed int64 capacity")
    if any(not np.isfinite(x) or not 0 < x < 0.5 for x in errors):
        raise ValueError("epsilon must be in (0,0.5)")
    if any(not np.isfinite(x) or not 0 < x <= 1 for x in mutations):
        raise ValueError("mu must be in (0,1]")
    if any(not np.isfinite(x) or x < 1 for x in ratios):
        raise ValueError("b/c must be finite and >=1")
    conditions = []
    for network, epsilon in itertools.product(networks, errors):
        network_bc = list(ratios)
        if args.profile == "paper" and network == "regular" and args.bc is None:
            network_bc += spec.get("regular_extra_bc", [])
        for bc, mu in itertools.product(network_bc, mutations):
            if (args.profile == "paper" and bc not in spec["bc"]
                    and network != "regular"):
                continue
            # Exactly the same keys and float types used in the original HPC
            # condition hash. Adding a field here would change the random stream.
            condition = dict(network=network, epsilon=float(epsilon), bc=float(bc),
                             delta=float(config["delta"]), mu=float(mu), cost=float(config["cost"]))
            if args.model == "pairwise":
                conditions.extend(dict(condition, opponent_id=i) for i in opponents)
            else:
                conditions.append(condition)
    if not conditions:
        raise ValueError("No matching paper conditions")
    return dict(model=args.model, profile=args.profile, conditions=conditions, batches=batches,
                steps=steps, burn_in=burn, blocks=blocks, replicate_start=args.replicate_start,
                cost=float(config["cost"]), delta=float(config["delta"]),
                payoff_refresh_steps=int(config["payoff_refresh_steps"]))


def run(plan, args):
    module = importlib.import_module(MODULES[args.model])
    graphs = module.make_graphs()
    settings = dict(grid_size=6, cost=plan["cost"],
                    payoff_refresh_steps=plan["payoff_refresh_steps"])
    runtime = dict(python=platform.python_version(), numpy=np.__version__,
                   numba=numba.__version__, platform=platform.platform())
    code = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in ["simulate.py", MODULES[args.model] + ".py"]}
    manifest = dict(schema=1, plan=plan, runtime=runtime, source_sha256=code,
                    graph_sha256={name: hashlib.sha256(graph.astype("<i8").tobytes()).hexdigest()
                                  for name, graph in graphs.items()})
    output = args.output or Path("results") / f"{args.model}_{args.profile}"
    with output_lock(output):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Output contains a different configuration/runtime/code; choose a new --output directory")
        else:
            atomic_json(manifest_path, manifest)
        summary = []
        for index, condition in enumerate(plan["conditions"]):
            tag = digest(condition)[:20]
            engine = module.Engine(graphs[condition["network"]], condition, settings)
            engine.compile()
            all_means, all_richness = [], []
            for batch in plan["batches"]:
                signature = dict(model=args.model, condition=condition, batch=batch,
                                 steps=plan["steps"], burn_in=plan["burn_in"], blocks=plan["blocks"],
                                 replicate_start=plan["replicate_start"], settings=settings,
                                 runtime=runtime, source_sha256=code,
                                 graph_sha256=manifest["graph_sha256"][condition["network"]])
                result_path = output / f"{tag}_{batch['name']}.npz"
                if result_path.exists():
                    with np.load(result_path, allow_pickle=False) as saved:
                        if str(saved["signature_json"]) != encoded(signature):
                            raise ValueError(f"Result signature mismatch: {result_path}")
                        means = saved["replicate_means"].copy()
                        richness = saved["replicate_coexistence_means"].copy()
                        if (means.shape != (batch["replicates"], engine.k)
                                or richness.shape != (batch["replicates"],)
                                or not np.isfinite(means).all() or not np.isfinite(richness).all()
                                or np.any(means < 0) or not np.allclose(means.sum(axis=1), 1)
                                or np.any(richness < 1) or np.any(richness > engine.k)):
                            raise ValueError(f"Invalid saved result: {result_path}")
                    print(f"{index+1}/{len(plan['conditions'])} {condition} {batch['name']}: already complete", flush=True)
                else:
                    count_sums = np.zeros((batch["replicates"], plan["blocks"], engine.k), np.int64)
                    richness_sums = np.zeros((batch["replicates"], plan["blocks"]), np.int64)
                    finals = np.zeros((batch["replicates"], engine.k), np.int64)
                    checkpoint_paths = []
                    for position in range(batch["replicates"]):
                        replicate = plan["replicate_start"] + position
                        checkpoint = output / "checkpoints" / f"{tag}_{batch['name']}_{replicate:05d}.npz"
                        checkpoint_paths.append(checkpoint)
                        rng = module.condition_seed(batch["seed"], condition, replicate)
                        sums, counts, coexistence = module.run_checkpointed_trajectory(
                            engine, rng, plan["steps"], plan["burn_in"], plan["blocks"], checkpoint,
                            dict(signature, replicate=replicate), args.chunk_steps, args.checkpoint_seconds)
                        count_sums[position], finals[position], richness_sums[position] = sums, counts, coexistence
                        print(f"{index+1}/{len(plan['conditions'])} {condition} {batch['name']}: "
                              f"trajectory {position+1}/{batch['replicates']} complete", flush=True)
                    means = count_sums.sum(axis=1) / (engine.n * plan["steps"])
                    richness = richness_sums.sum(axis=1) / plan["steps"]
                    module.atomic_npz(result_path, signature_json=np.array(encoded(signature)),
                                      condition_json=np.array(encoded(condition)), strategies=engine.strategies,
                                      replicate_ids=np.arange(plan["replicate_start"],
                                                              plan["replicate_start"]+batch["replicates"]),
                                      replicate_means=means, replicate_coexistence_means=richness,
                                      block_count_sums=count_sums, block_richness_sums=richness_sums,
                                      final_counts=finals)
                    # The complete, atomic batch result now contains all recorded
                    # sums. Discard its redundant restart files to avoid accumulating
                    # hundreds of thousands of checkpoints in a full paper run.
                    for checkpoint in checkpoint_paths:
                        checkpoint.unlink()
                all_means.append(means)
                all_richness.append(richness)
            merged = np.concatenate(all_means)
            for strategy_id, values in enumerate(merged.T):
                row = dict(condition, strategy_id=strategy_id,
                           strategy=encoded(engine.strategies[strategy_id].tolist()),
                           trajectories=len(values), mean_abundance=float(values.mean()),
                           mean_coexistence=float(np.concatenate(all_richness).mean()))
                if args.model == "pairwise":
                    row["role"] = "opponent" if strategy_id == 0 else "ALLC"
                summary.append(row)
        temporary = output / "abundances.csv.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        os.replace(temporary, output / "abundances.csv")
        print(f"Complete: {output.resolve()}")


def main():
    parser, args = arguments()
    try:
        config = json.loads(args.config.read_text())
        plan = make_plan(args, config)
        trajectories = len(plan["conditions"]) * sum(batch["replicates"] for batch in plan["batches"])
        print(f"{args.model}: {len(plan['conditions'])} conditions; {trajectories:,} trajectories total; "
              f"{plan['burn_in']:,} burn-in + {plan['steps']:,} sampled node updates each.")
        if args.list:
            print(json.dumps(plan, indent=2))
        else:
            run(plan, args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
