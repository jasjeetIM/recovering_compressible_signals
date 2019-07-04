## Repo Overview
This repo contains experimental code for the paper "Recovery Guarantees for Compressible Signals with Adversarial Noise".

## Code Overview
- /models: Core codebase for network setup, training, one pixel attack, and other  utilities.
- /cleverhans: Branch of the Cleverhans library used in conducting the CW, JSMA, and DF attacks. 
- /notebooks: Jupyter notebooks for each experiment. Experiments for Section 8.1 are named Theory-* 

## Requirements
- python 2.7.12
- keras 2.2.4, with GPU support
- tensorflow 1.8.0, with GPU support
- numpy(1.16.14), scipy(1.2.2), bottleneck(1.2.1), cvxpy(1.0.24), PIL(5.1.0), jupyter(4.4.0)

## Usage
We have created a notebook for every experiment in the paper. In order to reproduce or modify experiments, follow the relevant notebook.
