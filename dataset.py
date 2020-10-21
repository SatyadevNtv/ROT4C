import numpy as np
from scipy.io import loadmat

import utils

from IPython import embed

def load_train_test_data(args):
    data = loadmat(args.data_path)
    Xtr, Ytr = data['Xtr'], data['Ytr']
    Xte, Yte = data['Xte'], data['Yte']
    return Xtr, Ytr, Xte, Yte
