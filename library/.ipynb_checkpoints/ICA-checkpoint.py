import numpy as np
import matplotlib.pyplot as plt

from time import time
from .BASE import BASE
from numpy.linalg import inv
from sklearn.decomposition import FastICA

class ICA(BASE):
    """
    ICA anomaly detection class
    Developer: Sangjun Han (sjun.han@lgsp.co.kr)
    Date: 2020-03-29

    Parameters
    ----------
    num_comp : The number of ICA component (less than sensors number)
    max_iter : Max iteration of ICA
    """

    def __init__(self, num_comp=None, max_iter=3000):
        self.num_comp = num_comp
        self.max_iter = max_iter

    def cumSum(self):
        """
        Explain cumulative sum of weights
        """

        explain = np.sum(np.square(self.W), axis=1)
        explain_ratio = explain / np.sum(explain)

        return explain_ratio

    def getMetrics(self, X):
        # 1. I_dominant_squared
        S_d = np.dot(self.W[:self.num_comp, :], X.T)
        I_d = np.diag(np.dot(S_d.T, S_d))

        # 2. I_error_squared
        S_e = np.dot(self.W[self.num_comp:, :], X.T)
        I_e = np.diag(np.dot(S_e.T, S_e))

        # 3. SPE
        Q_inv = inv(self.Q)
        B_d = np.dot(self.W[:self.num_comp, :], Q_inv).T
        approx = np.dot(Q_inv, np.dot(B_d, S_d)).T
        error = X - approx # X - X^
        SPE = np.diag(np.dot(error, error.T)) # error^2

        return I_d, I_e, SPE

    def fit(self, x_train):
        """
        Fit the ICA model according to the given training data

        Parameters
        ----------
        x_train : n x p (time_samples, num_sensors)

        Returns
        -------
        I_d : Metrics for principal component (n x 1)
        I_e : Metrics for non-principal component (n x 1)
        SPE : Metrics for reconstruction error (n x 1)
        """

        start_time = time()
        n, p = x_train.shape[0], x_train.shape[1]

        ica = FastICA(n_components=p, whiten=True, max_iter=self.max_iter)
        S = ica.fit_transform(x_train)

        # required matrix
        self.W = ica.components_ # unmixing matrix
        self.A = ica.mixing_ # mixing matrix
        self.Q = ica.whitening_ # whitening matrix

        # sorted by sum of l2 norm
        sorted_idx = np.argsort(np.sum(np.square(self.W), axis=1))[::-1]
        self.W = self.W[sorted_idx, :]

        # explain cumsum of weights
        print('Cumsum of weights :', np.sum(self.cumSum()[:self.num_comp]))

        # get I_d, I_e, SPE
        I_d, I_e, SPE = self.getMetrics(x_train)

        print('It takes %0.2f sec for training model' % (time()-start_time))

        return I_d, I_e, SPE
    
    def predict(self, x_test):
        """
        Predict according to the PCA model

        Parameters
        ----------
        x_test : n x p (time_samples, num_sensors)

        Returns
        -------
        I_d : Metrics for principal component (n x 1)
        I_e : Metrics for non-principal component (n x 1)
        SPE : Metrics for reconstruction error (n x 1)
        """

        # get I_d, I_e, SPE
        I_d, I_e, SPE = self.getMetrics(x_test)

        return I_d, I_e, SPE
