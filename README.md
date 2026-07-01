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

The remaining files are subfunctions or data needed to run the above codes.

The theoretical_calculation_Python folder contains the following Python scripts, corresponding one-to-one to the MATLAB main codes:

--Main_category_size_5_py: Python version of Main_category_size_5.

--Main_star_transition_probability_py: Python version of Main_star_transition_probability.

--Main_fan_transition_probability_py: Python version of Main_fan_transition_probability.

All subfunctions are integrated in network_calc.py. The data files (n_5.mat and n_5_coords.mat) are used for running Main_category_size_5_py.

NOTE
-----------
To run the code, make sure that all files are in the same folder.
