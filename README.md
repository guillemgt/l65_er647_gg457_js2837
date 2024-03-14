# A New Family of GNNs using State Space Models

Evan Rex, Jad Sbai & Guillem Tarrach, A New Family of GNNs using State Space Models for Geometric Deep
Learning (L65), University of Cambridge.

## Abstract
Message Passing Neural Networks suffer from several problems when it comes
to modelling long-range interactions between nodes due to their local nature.
Recently, Space State Models (SSM) such as Mamba have been shown to be
capable of modeling extremely long-range interactions for sequential data. In this
report, we present a comprehensive study of different ways to incorporate SSMs to
handle graph data. We highlight difficulties faced by these approaches, as well as
indicate research directions to address these barriers.

## Codebase

This code base is branched off of the [codebase](https://github.com/bowang-lab/Graph-Mamba) for Wang, Chloe, et al. "Graph-mamba: Towards long-range graph sequence modeling with selective state spaces." arXiv preprint arXiv:2402.00789 (2024). 

Our additions include several new `global_model_type`s to the GraphGPS.layer GPSLayer class: 
- EfficientSubgraph_Mamba_L65
- SharedMambaL65
- MambaL65
- GraphSSML65
- MultiMambaL65
- MeanL65
- ConvL65

We use slurm to launch our experiments, sbatch scripts for this can be found in slurm_sweep_scripts and slurm_scripts. The base configuration files for the 3 datasets can be found in configs/Experiments.

## Installation instructions:

```
conda create --name L65 python=3.10.12
conda activate L65
```
Install torch: https://pytorch.org/get-started/locally/, e.g. `pip3 install torch torchvision torchaudio`.
Needs to be done before requirements.txt install.

```
pip install -r requirements.txt
```
Now install causal_conv1d and mamba-ssm:
```
git clone https://github.com/Dao-AILab/causal-conv1d.git
```
Edit `causal-conv1d/setup.py` by changing
```
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_70,code=sm_70")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
```
into
```
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_60,code=sm_60")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_70,code=sm_70")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
```
and run
```
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install causal-conv1d/
```
and also
```
git clone https://github.com/state-spaces/mamba.git
```
Edit `mamba/causal-conv1d/setup.py` by changing
```
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_70,code=sm_70")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
```
into
```
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_60,code=sm_60")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_70,code=sm_70")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
```
and run
```
MAMBA_FORCE_BUILD=TRUE pip install mamba/
```

If you want to quickly test if mamba is installed correctly:
```
python mamba-test.py
```
Then delete `causal_conv1d/` and `mamba/`:
```
rm -rf causal-conv1d/
rm -rf mamba/
```

Finally, add the following packages:
```
pip install rdkit deepspeed local_attention axial_positional_embedding performer_pytorch ogb torch_scatter yacs
```

## Setting up to run on HPC:
Run the following commands:

module purge

module load rhel8/default-amp

module unload miniconda/3

module load cuda/11.8

Run `nvcc -V` to verify that your cuda version is 11.8

Follow instructions from previous section to create a conda environment and activate it. 

Set the $CUDA_HOME environment variable to the 11.8 cuda environment:

conda env config vars set CUDA_HOME=/path/to/cuda-11.8

Note that the path to cuda11.8 can be found by running `module show cuda/11.8` and then copying what comes after CUDA_PATH. So the command would likely be:
conda env config vars set CUDA_HOME=/usr/local/software/cuda/11.8

Run conda activate L65 again.

Now follow rest of the install instructions. But make sure to install correct torch for cuda=11.8. e.g.:`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
