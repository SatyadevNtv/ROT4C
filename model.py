import time
import tensorflow as tf
import gc

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from scipy.spatial.distance import cdist
from scipy.io import loadmat
from sklearn.preprocessing import normalize

import algo as algorithms
import utils

from IPython import embed

def prediction_softmax(x, theta):
    """
    Predict using softmax function
    """
    return tf.nn.softmax(tf.matmul(x, theta))


class Model():
    """
    Implements a 1-layer Logistic Regression model
    """

    def __init__(
            self,
            args,
            Xtr,
            Ytr,
            Xte,
            Yte,
    ):
        self.options = args
        self.Xtr = Xtr
        self.Xte = Xte
        self.Ytr = Ytr
        self.Yte = Yte
        self.num_features, self.num_classes = self.Xtr.shape[1], self.Ytr.shape[1]
        self.current_epoch = -1
        self.prediction = prediction_softmax
        if self.options.load_model_from is not None:
            self.W = tf.Variable(np.loadtxt(self.options.load_model_from), name="weight")
            print(f"Initial AUC is")
            self.log_accuracy()
        else:
            self.W = tf.Variable(np.ones((self.num_features, self.num_classes)), name="weight")

        wembs = loadmat(self.options.w2v_embs)
        self.Xs, self.Xt, self.emb_dim = wembs["Xs"], wembs["Xt"], 300

        if self.options.normalize:
            print(f"Normalizing word embeddings ..")
            self.Xs = normalize(self.Xs)
            self.Xt = normalize(self.Xt)

        self.optimizer = tf.optimizers.SGD(learning_rate=self.options.learning_rate)
        if self.options.lossf == "W22":
            C = cdist(self.Xs, self.Xt)**2
            self.lossfx = algorithms.W22(C, args, self.num_features, self.num_classes)
        elif self.options.lossf == "RbOT":
            PArray = []
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    tmp = self.Xs[i].reshape(self.options.d1, self.options.r) - self.Xt[j].reshape(self.options.d1, self.options.r)
                    PArray.append(tmp.T.dot(tmp).reshape(self.options.r**2, 1))
            P = np.hstack(PArray)
            del PArray
            self.lossfx = algorithms.RbOT(P, args, self.num_features, self.num_classes)
        else:
            raise ValueError

    def optimize(self, X, Y):
        with tf.GradientTape() as g1:
            loss2 = 0.0005 * tf.norm(self.W)**2
        gradients2 = g1.gradient(loss2, [self.W])

        if self.options.rbot_solver == "FW":
            loss1, gradients1 = self.lossfx.gradient(X, self.prediction(X, self.W).numpy(), Y.numpy())
        elif self.options.rbot_solver == "Manf":
            loss1, gradients1 = self.lossfx.gradient_manf(X, self.prediction(X, self.W).numpy(), Y.numpy())
        else:
            raise ValueError
        gradients1 = [gradients1]

        finalGrad = [gradients1[0] + gradients2[0]]

        # Update W
        self.optimizer.apply_gradients(zip(finalGrad, [self.W]))
        return loss1, loss2, finalGrad


    def log_accuracy(self, save=False):

        if save and self.options.save_model_to is not None:
            print(f"Saving weights to {self.options.save_model_to} ...")
            np.savetxt(f'{self.options.save_model_to}/W{self.options.end_epoch}.txt', self.W.numpy())

        # Compute on CPU to reduce GPU memory usage.
        # If enough memory is available, this can be toggled.
        with tf.device('/CPU:0'):
            preds = self.prediction(self.Xte, self.W.numpy())
            gts = self.Yte
            print(f"Test Accuracy (AUC): {roc_auc_score(gts, preds)}")


    def run(self):

        for epoch in range(self.options.start_epoch, self.options.end_epoch + 1):
            train_data=tf.data.Dataset.from_tensor_slices((self.Xtr, self.Ytr))
            train_data=train_data.shuffle(self.Xtr.shape[0]).batch(self.options.batch_size)

            print(f"Epoch {epoch}")
            epoch_num = epoch

            total_loss_W22 = 0
            total_loss2 = 0
            for batchX, batchY in train_data:
                loss_batch_w22, loss_batch2, gradW = self.optimize(batchX, batchY)
                total_loss_W22 += loss_batch_w22
                total_loss2 += loss_batch2
            if epoch % 10 == 0:
                self.log_accuracy(save=True)

        self.log_accuracy(save=True)

