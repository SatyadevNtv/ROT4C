Robust Optimal Transport for multi-class/label learning.

## Overview

This repo implements different Robust Optimal Transport fomulations as discussed in the paper.

## Requirements

- Python (3.6+)

Install requirements from `requirements.txt`

## Usage

Checkout `--help` of `rb_ot.py` for detailed usage.

```sh
python rb_ot.py --help
```

### Sample

A helper bash script is provided that runs the multi-class learning setup on a sample of AwA
data in `./data/` folder

```sh
Expected AUC: 0.903
```

**NOTE**

For GPU usage, checkout `algo.py` (line no. 1) and `sinkhorn_gpu.py` (line no. 12)
