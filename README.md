# Embodied decisions as active inference

<p align="center">
  <img src="/reference/images/env.png">
</p>

This is the project related to the papers [Embodied decisions as active inference](https://doi.org/10.1101/2024.05.28.596181) and [Learning and embodied decisions in active inference
](https://www.biorxiv.org/content/10.1101/2024.08.18.608439v1). In this study, we introduce a novel modeling approach to explore embodied decision-making, where decisions and actions occur simultaneously in dynamic environments. Our model offers a new perspective on how decisions are influenced by the actions taken, highlighting the importance of considering motor control as an integral part of decision processes.

Video simulations are found [here](https://priorelli.github.io/projects/7_embodied_decisions/).

Check [this](https://priorelli.github.io/projects/) and [this](https://priorelli.github.io/blog/) for additional projects and guides.

## HowTo

### Start the simulation

The simulation can be launched through *main.py*, either with the option `-m` for manual control, `-e` for the simulation in Figure 2, `-u` for the simulation in Figure 3, `-d` for the simulation in Figure 4, `-t` for the simulation in Figure 5, `-s` for the simulation in Figure 6, `-a` for the simulation in Figure 8, `-l` for the simulation in Figure 9, `-k` for the learning simulation in [Learning and embodied decisions in active inference
](https://www.biorxiv.org/content/10.1101/2024.08.18.608439v1), or `c` for a custom configuration specified in *config.py*. If no option is specified, the custom simulation will be launched. For the manual control simulation, the arm can be moved with the keys `Z`, `X`, `LEFT`, `RIGHT`, `UP` and `DOWN`.

Plots can be generated through *plot.py*, either with the option `-e` for Figure 2, `-u` for Figure 3, `-t` for Figure 5, `-s` for Figure 6, `-a` for Figure 8, `-l` for Figure 9, or `-v` for generating a video of the simulation.

### Advanced configuration

More advanced parameters can be manually set from *config.py*. Custom log names are set with the variable `log_name`. The number of trials and steps can be set with the variables `n_trials` and `n_steps`, respectively.

The parameter `n_tau` specifies the sampling time window used for evidence accumulation.

The parameter `n_cues` specifies number of the cues, `cue_prob` the cue probability for the first target, `cue_sequence` a custom sequence of cues.

The agent configuration is defined through the dictionary `joints`. The value `link` specifies the joint to which the new one is attached; `angle` encodes the starting value of the joint; `limit` defines the min and max angle limits.

Useful trajectories computed during the simulations are stored through the class `Log` in *environment/log.py*.

Note that all the variables are normalized between -1 and 1 to ensure that every contribution to the belief updates has the same magnitude.

## Required libraries

matplotlib==3.10.1

numpy==2.2.4

pandas==2.2.3

pyglet==2.1.3

pymunk==6.11.1

scipy==1.15.2

seaborn==0.13.2

torch==2.6.0
