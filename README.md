# Constrained Model-Based Policy Optimization

This repository contains a model-based version of Constrained Policy Optimization (Achiam et al.).

<p align="center">
	<!-- <img src="https://drive.google.com/uc?export=view&id=1DcXi5wY_anmtlNeIErl1ECgKGsGi4oR1" width="80%"> -->
	<img src="https://drive.google.com/uc?export=view&id=1DcXi5wY_anmtlNeIErl1ECgKGsGi4oR1" width="80%">
</p>

## Prerequisites
1. Mujoco 2.0 has to be installed and a workin license is required. 
2. Instructions on installing mujoco can be found here [Mujoco-Py](https://github.com/openai/mujoco-py)
3. We use conda environments for installs, please refer to [Anaconda](https://docs.anaconda.com/anaconda/install/) for instructions. 

## Installation
1. Clone this repository
```
git clone https://github.com/anyboby/ConstrainedMBPO.git
```
2. Create a conda environtment and install cmbpo from egg
```
cd ConstrainedMBPO
conda env create -f environment/gpu-env.yml
conda activate cmbpo
pip install -e .
```

## Usage
To start an experiment with cmbpo, run 
```
cmbpo run_local examples.development --config=examples.config.antsafe.cmbpo_uc --policy=cpopolicy --gpus=1 --trial-gpus=1
```

`-- config` specifies the configuration for an environment (here: antsafe)   
`-- policy` specifies the policy to use, as of writing cmbpo only works on a cpo policy   
`-- gpus` specifies the number of gpus to use   

As of writing, only local running is supported. For further options, refer to the ray experiment specifications. 

#### Testing on new environments
Different environments can be tested by creating a config file in the [config](examples/config/) directory. The algorithm does not learn termination functions and unless otherwise specified will not terminate. To create a manual terminal function, do so by creating a file with the lower-case name of the environment under 'mbpo/static'.

#### Logging
A wide range of data is logged automatically in tensorboard. The corresponding files and checkpoints are stored in **~/ray\_mbpo/**. To view the tensorboard logs run
```
tensorboard --logdir ~/ray\_mbpo/<env>/<seed_dir>
```
#### Hyperparameters
The hyperparameter list as of now is a bit messy. The hyperparameters for cmbpo are specified in the config file (e.g., [config_file](examples/config/antsafe/cmbpo_uc.py)).
The main parameters for cmbpo are
```
'rollout_mode':'uncertainty',
'max_uncertainty_c':4.0,
'max_uncertainty_rew':3.5,
'max_tddyn_err':.07,
```
The rollout mode specifies when to terminate rollouts. The `'uncertainty'` rollout mode terminates rollouts based on how much the uncertainty of model-predictions grows compared to the first prediction. Other options are `'iv'`, an inverse-variance weighting of rollout lengths by the variance of rewards, and `'schedule'`, a fixed schedule of rollout-lengths specified by `'rollout_schedule'`.

The `'max_uncertainty'` only applies to rollout modes `'uncertainty'` and `'iv'` where this parameter limits the ratio of predictitive uncertainty to the initial predictions (for rew and cost seperately, but in our experiments similar values for both worked well). 

`'max_tddyn_err'` refers to the maximum temporal difference dynamics error and specifies how much we rely on the model. This variable is calculated as mean variance among ensemble predictions, normalized by the base variance of targets. Assuming we specifiy this maximum uncertainty to be 0.05 and we measure an average uncertainty of 0.1 among our ensemble predictions, the algorithm will train on a dataset consisting of half model-generated samples and half real samples (resulting in an average error of .05).

## Acknowledgments
This repository also contains several more algorithms including SAC from [Tuomas Haarnoja](https://scholar.google.com/citations?user=VT7peyEAAAAJ&hl=en), [Kristian Hartikainen's](https://hartikainen.github.io/) [softlearning](https://github.com/rail-berkeley/softlearning) codebase, [Kurtland Chua's](https://kchua.github.io/) dynamics ensemble in [PETS](https://github.com/kchua/handful-of-trials), and CPO from [Joshua Achiam](https://github.com/openai/safety-starter-agents). 
