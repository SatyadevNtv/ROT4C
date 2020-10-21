import argparse
import numpy as np
import tensorflow as tf

import utils
import dataset
from model import Model

from IPython import embed

def parse_cmdline():
    parser = argparse.ArgumentParser(description="Robust Optimal Transport for Classification")
    parser.add_argument('--data_path', type=str, help="Path to train, test split data")
    parser.add_argument('--lossf', default="RbOT", type=str, choices=["W22", "RbOT"], help="Loss function for the predictions")
    parser.add_argument('--verbose', default=0, type=int, help="Verbose")
    parser.add_argument('--w2v_embs', type=str, help="Path to Word2Vec embeddings")
    parser.add_argument('--normalize', default=True, type=bool, help="Normalize word embeddings")


    regularizer_group = parser.add_argument_group("Regularizer (on Mahalanobis metric) parameters")
    regularizer_group.add_argument('--reg_type', default="pnorm", type=str, choices=["pnorm", "kl", "ds"], help="Type of regularizer")
    regularizer_group.add_argument('--pnorm_k', default=1, type=int, help="`k` for elem-wise p-norm regularizer")
    regularizer_group.add_argument('--tikhr', default=1, type=float, help="tikhonov regularizer for KL, DS")
    regularizer_group.add_argument('--minit', default="eye", choices=["eye", "ones"], type=str, help="M_0 for KL")

    optimizer_group = parser.add_argument_group("Optimization params")
    optimizer_group.add_argument('--rbot_solver', default="FW", choices=["FW"], type=str, help="RbOT solver")
    optimizer_group.add_argument('--save_model_to', default=None, type=str, help="dir path to save the model")
    optimizer_group.add_argument('--load_model_from', default=None, type=str, help="Path to load the model weights")
    optimizer_group.add_argument('--batch_size', default=150, type=int, help="Batch size for SGD")
    optimizer_group.add_argument('--learning_rate', default=1, type=float, help="Learning rate for SGD")
    optimizer_group.add_argument('--start_epoch', default=0, type=int, help="Starting epoch")
    optimizer_group.add_argument('--end_epoch', default=200, type=int, help="Ending epoch")
    optimizer_group.add_argument('--fw_iters', default=1, type=int, help="Max FW iterations")
    optimizer_group.add_argument('--fw_mu', default=0.2, type=float, help="regularizer used in FW")
    optimizer_group.add_argument('--sinkhorn_reg', default=0.02, type=float, help="regularizer used in sinkhorn")
    optimizer_group.add_argument('--sinkhorn_iters', default=10, type=int, help="Max Sinkhorn iterations")

    dr_group = parser.add_argument_group("Dimensionality reduction params")
    dr_group.add_argument('--d1', default=30, type=int)
    dr_group.add_argument('--r', default=10, type=int)

    args = parser.parse_args()
    if args.verbose:
        print(f"Current arguments: {args}")
    return args


def vprint(msg):
    if VERBOSE:
        print(msg)



if __name__ == "__main__":
    np.random.seed(1)
    tf.random.set_seed(1)

    args = parse_cmdline()
    global VERBOSE
    VERBOSE = args.verbose
    Xtr, Ytr, Xte, Yte = dataset.load_train_test_data(args)
    vprint(f"Data shape: {Xtr.shape}, {Ytr.shape}")


    model = Model(args, Xtr, Ytr, Xte, Yte)
    model.run()
