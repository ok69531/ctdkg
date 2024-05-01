# Preparing for CTDKG


## Linux

```bash
$ conda create -n ctdkg python==3.10.0 -y
$ conda activate ctdkg
$ pip install -r requirements-linux.txt
```
Please follow the links below to install PyTorch, torch-scatter, and torch-sparse with other CUDA versions:

- https://pytorch.org/
- https://data.pyg.org/whl/


## Silicon-mac

```bash
$ conda create -n ctdkg python==3.10.0 -y
$ conda activate ctdkg
$ pip install -r requirements-silicon.txt
```
