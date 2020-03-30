import numpy as np
import matplotlib.pyplot as plt

from time import time
from numpy.linalg import inv
from scipy.stats import norm, f

class PCA():
    """
    PCA anomaly detection class
    Developer: Sangjun Han (sjun.han@lgsp.co.kr)
    Date: 2020-03-29

    Parameters
    ----------
    num_comp : The number of PCA component (less than sensors number)
    alpha_tsq : The confidence of T square
    alpha_spe : The confidence of SPE
    """

    def __init__(self, num_comp=10, alpha_tsq=0.99, alpha_spe=0.99):
        self.num_comp = num_comp
        self.alpha_tsq = alpha_tsq
        self.alpha_spe = alpha_spe

    def cumVar(self):
        """
        Explain cumulative variance of Eigen matrix
        """

        S_var = np.square(self.S) / (self.n - 1)
        return S_var / np.sum(S_var)

    def getThreshold(self):
        """
        Tsq is on F distribution, and SPE is on Q distribution
        It depends on the confidence of user setting
        """

        # Tsq on F distribution
        dfn = self.num_comp
        dfd = self.n - self.num_comp
        F = f.ppf(q=self.alpha_tsq, dfn=dfn, dfd=dfd)
        Tsq_thres = (dfn * (self.n - 1) * F) / dfd

        # SPE on Q distribution
        theta_1 = np.sum(self.S[self.num_comp:]**1)
        theta_2 = np.sum(self.S[self.num_comp:]**2)
        theta_3 = np.sum(self.S[self.num_comp:]**3)

        C = norm.ppf(self.alpha_spe)
        h0 = 1 - ((2 * theta_1 * theta_3) / (3 * (theta_2**2)))

        SPE_1 = (C * np.sqrt(2 * theta_2 * (h0**2))) / theta_1
        SPE_2 = (theta_2 * h0 * (h0 - 1)) / (theta_1**2)
        SPE_thres = theta_1 * ((SPE_1 + 1 + SPE_2)**(1 / h0))

        return Tsq_thres, SPE_thres

    def fit(self, x_train):
        """
        Fit the PCA model according to the given training data

        Parameters
        ----------
        x_train : n x p (time_samples, num_sensors)

        Returns
        -------
        Tsq_thres : Anomaly threshold for Tsq
        SPE_thres : Anomaly threshold for SPE
        """

        start_time = time()
        n, p = x_train.shape[0], x_train.shape[1]
        x_cov = np.dot(x_train.T, x_train) / (n - 1) # get covariance
        U, S, V = np.linalg.svd(x_cov, full_matrices=True) # X = USV^T

        # keep variable
        self.n = n # time_samples
        self.p = p # num_sensors
        self.U = U # transforming matrix
        self.S = S # Eigen matrix

        # explain variance of Eigen matrix
        print('Cumsum of variance :', np.sum(self.cumVar()[:self.num_comp]))

        # transform to PCA space
        T = np.dot(x_train, U[:, :self.num_comp])

        # Tsq statistics
        S_inv = inv(np.diag(S[:self.num_comp]))
        Tsq = np.diag(np.dot(np.dot(T, S_inv), T.T))

        # SPE statistics
        E = np.dot(x_train, U[:, self.num_comp:])
        SPE = np.diag(np.dot(E, E.T))

        # get threshold of Tsq and SPE
        Tsq_thres, SPE_thres = self.getThreshold()

        print('It takes %0.2f sec for training model' % (time()-start_time))

        return Tsq_thres, SPE_thres
    
    def predict(self, x_test):
        """
        Predict according to the PCA model

        Parameters
        ----------
        x_test : n x p (time_samples, num_sensors)

        Returns
        -------
        Tsq : Tsq (n x 1)
        SPE : SPE (n x 1)
        """

        # transform to PCA space
        T = np.dot(x_test, self.U[:, :self.num_comp])

        # Tsq statistics
        S_inv = inv(np.diag(self.S[:self.num_comp]))
        Tsq = np.diag(np.dot(np.dot(T, S_inv), T.T))

        # SPE statistics
        E = np.dot(x_test, self.U[:, self.num_comp:])
        SPE = np.diag(np.dot(E, E.T))

        return Tsq, SPE