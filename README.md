# RobustStrategy

DESCRIPTION
-----------

Code for paper "Robust strategies for iterated games on graphs".

The code is provided for theoretical calculations (written in Matlab R2024a) and numerical simulations (written in Python 3.10).

A Python version for the theoretical calculations (written in Python 3.10) is included in the theoretical_calculation_Python folder.

FILE
-----------

--Main_category_size_5: determine the category of all connected graphs of size five. Running this code produces the category outputs shown in Fig. 2A and B.

--Main_regular_fixation_probability_simulation: simulate the fixation probability of a mutant $ALLD^{\epsilon}$ taking over a population of $ALLC^{\epsilon}$ in regular graphs. This code allows for the simulation of the fixation probability of a reactive strategy invading another by adjusting the parameters $f$, $g$, $p$, and $q$. It can also be applied to other population structures, such as star and fan graphs.

--Main_star_transition_probability: calculate the transition probability matrix among the four corner strategies ($ALLC^{\epsilon}$, $TFT^{\epsilon}$, $ALLD^{\epsilon}$, and $ATFT^{\epsilon}$) in a star graph.

--Main_fan_transition_probability: calculate the transition probability matrix among the four corner strategies ($ALLC^{\epsilon}$, $TFT^{\epsilon}$, $ALLD^{\epsilon}$, and $ATFT^{\epsilon}$) in a fan graph.

--f_calculation_tau: function to calculate the values of $\tau_1$, $\tau_2$, and $\tau_3$ for a given network.

--f_network_category: function to determine the category of a given network.

The remaining files are subfunctions or data files needed to run the above scripts.

The theoretical_calculation_Python folder contains the following Python scripts, corresponding one-to-one with the main MATLAB scripts:

--Main_category_size_5_py: Python version of Main_category_size_5.

--Main_star_transition_probability_py: Python version of Main_star_transition_probability.

--Main_fan_transition_probability_py: Python version of Main_fan_transition_probability.

All subfunctions are integrated in network_calc.py. The data files (n_5.mat and n_5_coords.mat) are used for running Main_category_size_5_py.

NOTE
-----------
Run the original MATLAB scripts from the repository root. Run the original Python theory scripts from `theoretical_calculation_Python/`, where their `.mat` input files are located. The model extensions below can be launched from the repository root.

MODEL EXTENSIONS
----------------

The `model_extensions/` folder adds memory-one robustness calculations and
strategy competition under recurrent mutation. These abundance simulations
complement the original single-mutant fixation-probability experiment.

| File | Purpose |
|---|---|
| `theory.py` | Memory-one payoffs, network coefficients, and robustness intervals |
| `simulate.py` | Shared local simulation entry point with checkpoint recovery |
| `simulate_reactive36.py` | Competition among 36 reactive strategies |
| `simulate_memory_one16.py` | Competition among 16 memory-one corner strategies |
| `simulate_pairwise.py` | ALLC versus each of the other 35 reactive strategies |
| `paper_config.json` | Paper parameter grids, run lengths, and seed batches |

### Install and run

Use Python 3.11. The root `requirements.txt` pins tested dependencies for both
the original Python scripts and the extensions. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python model_extensions/theory.py
python model_extensions/simulate.py reactive
python model_extensions/simulate.py memory
python model_extensions/simulate.py pairwise
```

On Windows, activate with `.venv\Scripts\activate`. The extensions need no
HPC scheduler or MATLAB. The first simulation compiles the numerical kernels.

The theory command calculates all 192 combinations of 16 memory-one corner
residents, two mutant sets (16 corners and a 9^4 grid including both endpoints),
three networks, and epsilon = 0.1, 0.3. Under first-order weak selection,
robustness intervals are obtained by intersecting linear invasion inequalities
with b/c > 1, not by scanning b/c.
Results are saved in `results/theory/`.

Each default simulation is a **short demonstration**: one parameter combination,
two trajectories, 1,000 burn-in updates and 100,000 sampled updates per trajectory.
These examples check execution and do not establish stationarity.

### Paper configurations

`--profile paper` selects 1,000,000 burn-in node updates followed by
**2,000,000,000 sampled node updates per trajectory**, recorded in 128 blocks.
Each parameter combination combines 2,000 independent trajectories:

| Model | Trajectories x base seed |
|---|---|
| Reactive | 500 x 20261006; 500 x 20261007; 1,000 x 20261012 |
| Memory-one | 1,000 x 20261006; 1,000 x 20261013 |
| Pairwise | 1,000 x 20261011; 1,000 x 20261015 |

All parameter grids are in `paper_config.json`, including the reactive regular
b/c = 45, 50 extension. Pairwise runs use the cycle, epsilon = 0.1, mu = 0.01,
and b/c = 3, 5, 8. To inspect one paper condition without running it:

```bash
python model_extensions/simulate.py reactive --profile paper --network cycle --epsilon 0.3 --mu 0.01 --bc 6 --list
```

Remove `--list` to run it. Omitting parameter filters selects the full design,
which requires substantial computation. Runs are serial. `--batch` selects
one seed batch; `--replicate-start` and `--replicates` select trajectory indices.
Use `--output` to give independent subsets separate output folders.
`--steps`, `--burn-in`, `--blocks`, and `--seed` allow custom runs; `--opponent`
selects a pairwise opponent (0..34). See `--help` for all options.

### Conventions and outputs

The simulations use N = 50, c = 1, delta = 0.01 and the fixed cycle, degree-four
regular graph, and complete graph. Strategies are initially represented as
evenly as possible and randomly assigned to nodes (25 of each in pairwise runs).
Each update selects a node uniformly: with probability mu, a strategy is drawn
uniformly from the entire strategy set, including its current strategy;
otherwise a neighbor is copied in proportion to 1 + delta times its mean
payoff, calculated before replacement. Self-interactions are excluded.
A step is one node update, not a population sweep.

Reactive p and q each take six evenly spaced effective values in
[epsilon,1-epsilon]. Memory-one corner coordinates follow CC, CD, DC, DD,
with 0 denoting epsilon and 1 denoting 1-epsilon. No additional error is applied.
Pairwise opponent IDs enumerate the reactive grid with p outermost and q
innermost; ID 35 is ALLC and is excluded as an opponent.

Outputs under `results/` include trajectory means, integer block sums, final
counts, and coexistence counts in NPZ files. `abundances.csv` gives pooled mean
abundances as fractions (0..1); pairwise columns are **opponent, ALLC**.
`manifest.json` records the actual parameters, runtime versions, and code hashes.
Rerun the same command and output folder to resume; completed results are reused.
Each folder permits only one writer. Numerical behavior can vary across library
versions or hardware.

The kernels were checked against the production implementations, and theoretical
intervals against the reference calculations. Separate MCB, empirical threshold
estimation, and final figure-styling scripts are outside this module's scope.

