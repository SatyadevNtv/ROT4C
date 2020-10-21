#!/bin/bash

set -x
set -euo pipefail

python -W ignore rb_ot.py --verbose 1 --data_path ./data/awa_10_pc.mat --w2v_embs ./data/awa_wembs.mat
