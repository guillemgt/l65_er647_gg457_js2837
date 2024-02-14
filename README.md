# l65_er647_gg457_js2837

## Installation instructions:

```
conda create --name L65 python=3.10.12
conda activate L65
```
Install torch: https://pytorch.org/get-started/locally/
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
MAMBA_FORCE_BUILD=TRUE pip install mamba/
```

If you want to quickly test if mamba is installed correctly:
```
python mamba-test.py
```