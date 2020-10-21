## Robust Optimal Transport for Classification (ROT4C)

This repo implements different Robust Optimal Transport formulations as discussed in:

Pratik Jawanpuria, N T V Satya Dev, Bamdev Mishra. [Efficient robust optimal transport: formulations and algorithms](https://arxiv.org/abs/2010.11852), arXiv preprint arXiv:2010.11852, 2020

## Requirements

- Python (3.6+)

Install requirements from `requirements.txt`

## Usage

Checkout `--help` of `rot4c.py` for detailed usage.

```sh
python rot4c.py --help
```

### Sample

A helper bash script is provided that runs the multi-class learning setup on a sample of AwA
data in `./data/` folder

```sh
Expected AUC: 0.903
```

**NOTE**

For GPU usage, checkout `algo.py` (line no. 1) and `sinkhorn_gpu.py` (line no. 12)
