from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from sklearn.base import clone
from sklearn.metrics import zero_one_loss


class OOBaggingClassifier:
    def __init__(self, base_estimator, n_estimators=200):
        """
        Parameters
        ----------
        base_estimator: a probabilistic classifier that implements the predict_proba
        function, such as DecisionTreeClassifier
        n_estimators: the maximum number of estimators allowed.
        """
        self.base_estimator_ = base_estimator
        self.n_estimators = n_estimators
        self.estimators_ = []
        self.oob_errors_ = []

    def fit(self, X, y, random_state=None):
        """
        Train a model on the given training set.
                Parameters
                ----------
        X: an input array of shape (n_sample, n_features)
        X: an input array of shape (n_sample, n_features)
        es
        y: an array of shape (n_sample,) containing the classes for the input exampl
        """
        if random_state:
            np.random.seed(random_state)
        self.best_n = 0

        scores_oob = np.zeros((len(X), len(np.unique(y))))
        for i in range(self.n_estimators):
            estimator = clone(self.base_estimator_)
            # construct a bootstrap sample
            idx_bs = np.random.choice(len(X), size=len(X))
            X_bs = X[idx_bs]
            y_bs = y[idx_bs]
            # train on bootstrap sample
            estimator.fit(X_bs, y_bs)
            # predict on OOB examples
            idx_oob = np.setdiff1d(np.arange(len(X)), idx_bs)
            X_oob = X[idx_oob]
            p_oob = estimator.predict_proba(X_oob)
            # update class scores for OOB examples
            scores_oob[idx_oob] += p_oob
            # compute OOB error
            oob_pred = np.argmax(scores_oob, axis=1)
            oob_error = zero_one_loss(y, oob_pred)
            self.oob_errors_.append(oob_error)
            self.estimators_.append(estimator)
            if (self.best_n == 0) and (i >= 10) and (
                    np.mean(self.oob_errors_[-5:]) > np.mean(self.oob_errors_[-10:-5])):
                self.best_n = (i + 1)

    def errors(self, X, y):
        """
        Parameters ----------
        X: an input array of shape (n_sample, n_features)
        y: an array of shape (n_sample,) containing the classes for the input examples
        Returns ------
        error_rates: an array of shape (n_estimators,),
        with the error_rates[i] being the error rate of the ensemble consisting of the first (i+1) models.
        """
        scores = None
        error_rates = []
        for estimator in self.estimators_:
            p = estimator.predict_proba(X)
            if scores is None:
                scores = p
            else:
                scores += p
            preds = np.argmax(scores, axis=1)
            error_rates.append(zero_one_loss(y, preds))
        return error_rates

    def predict(self, X):
        """
        Parameters
        ----------
        X: an input array of shape (n_sample, n_features)
        Returns
        -------
        y: an array of shape (n_samples,) containig the predicted classes """

        probs = None
        for estimator in self.estimators_:
            p = estimator.predict_proba(X)
            if probs is None:
                probs = p
            else:
                probs += p
            return np.argmax(probs, axis=1)


X, y = load_digits(return_X_y=True)
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=42)

bagging = OOBaggingClassifier(DecisionTreeClassifier(), n_estimators=200)
bagging.fit(X_tr, y_tr, random_state=42)
print('# of selected models:', bagging.best_n)
error_rates = bagging.errors(X_ts, y_ts)
plt.plot(np.arange(len(bagging.oob_errors_)), bagging.oob_errors_, label='oob')
plt.plot(np.arange(len(bagging.oob_errors_)), error_rates, label='test')
plt.scatter([bagging.best_n], [0], marker='o')
plt.legend()
plt.show()

for seed in [1, 11]:
    print('Seed:', seed)
    bagging = OOBaggingClassifier(DecisionTreeClassifier(), n_estimators=100)
    bagging.fit(X_tr, y_tr, random_state=seed)
    print('# of selected models:', bagging.best_n)
    error_rates = bagging.errors(X_ts, y_ts)
    plt.plot(np.arange(len(bagging.oob_errors_)), bagging.oob_errors_, label='oob')
    plt.plot(np.arange(len(bagging.oob_errors_)), error_rates, label='test')
    plt.scatter([bagging.best_n], [0], marker='o')
    plt.legend()
    plt.show()
