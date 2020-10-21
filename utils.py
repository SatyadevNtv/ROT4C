import numpy as np
from ot.bregman import sinkhorn

from IPython import embed

def sinkhornAutoTune(a, b, C):
    entropic_reg = 1e-6
    while True:
        with np.errstate(over='raise', divide='raise'):
            try:
                print(f"Entropic regularizer: {entropic_reg}")
                gamma = sinkhorn(a, b, C, reg=entropic_reg, verbose=True)
                break
            except Exception as e:
                print(f"Caught Exception in SKnopp... {e}")
                entropic_reg = entropic_reg*2
