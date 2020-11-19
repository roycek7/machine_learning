import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement


def corrupt(X, y, outlier_ratio=0.1, random_state=None):
    random = check_random_state(random_state)

    n_samples = len(y)
    n_outliers = int(outlier_ratio * n_samples)

    W = X.copy()
    z = y.copy()

    mask = np.ones(n_samples).astype(bool)
    outlier_ids = random.choice(n_samples, n_outliers)
    mask[outlier_ids] = False

    W[~mask, 4] *= 0.1

    return W, z


class ENOLS:
    def __init__(self, n_estimators=500, sample_size='auto'):
        """
        Parameters
        ----------
        n_estimators: number of OLS models to train
        sample_size: size of random subset used to train the OLS models, default to 'auto'
            - If 'auto': use subsets of size n_features+1 during training
            - If int: use subsets of size sample_size during training
            - If float: use subsets of size ceil(n_sample*sample_size) during training
        """

        self.n_estimators = n_estimators
        self.sample_size = sample_size

    def fit(self, X, y, random_state=None):
        """
        Train ENOLS on the given training set.

        Parameters
        ----------
        X: an input array of shape (n_sample, n_features)
        y: an array of shape (n_sample,) containing the classes for the input examples

        Return
        ------
        self: the fitted model
        """

        # use random instead of np.random to sample random numbers below
        random = check_random_state(random_state)

        estimators = [('lr', LinearRegression())]

        if isinstance(self.sample_size, int):
            self.sample_size = 'reservoir_sampling'

        # add all the trained OLS models to this list
        self.estimators_lr, self.estimators_TSR, self.estimators_enols = [], [], []
        for i in range(self.n_estimators):
            samples = sample_without_replacement(n_population=random.choice([50, 100]),
                                                 n_samples=random.choice([10, 20]),
                                                 random_state=random_state, method=self.sample_size)

            X_train, y_train = [], []
            for i in samples:
                X_train.append(X[i]), y_train.append(y[i])

            reg = LinearRegression()
            reg.fit(np.array(X_train), np.array(y_train))

            tsr = TheilSenRegressor()
            tsr.fit(np.array(X_train), np.array(y_train))

            enol = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
            enol.fit(np.array(X_train), np.array(y_train))

            self.estimators_lr.append(reg), self.estimators_TSR.append(tsr), self.estimators_enols.append(enol)

        return self

    def predict(self, X, method='average'):
        """
        Parameters
        ----------
        X: an input array of shape (n_sample, n_features)
        method: 'median' or 'average', corresponding to predicting median and
            mean of the OLS models' predictions respectively.

        Returns
        -------
        y: an array of shape (n_samples,) containing the predicted classes
        """

        ols, ts_reg, enols = [], [], []
        for reg in self.estimators_lr:
            ols.append(reg.predict(X))

        for tsr in self.estimators_TSR:
            ts_reg.append(tsr.predict(X))

        for enol in self.estimators_enols:
            enols.append(enol.predict(X))

        if method == 'average':
            ols = np.average(ols, axis=0)
            enols = np.average(enols, axis=0)
            ts_reg = np.average(ts_reg, axis=0)
        else:
            ols = np.average(ols, axis=0)
            enols = np.median(enols, axis=0)
            ts_reg = np.median(ts_reg, axis=0)

        return ols, ts_reg, enols


if __name__ == '__main__':
    ensemble_ols, tsregressor, ordinaryleastsquare = [], [], []
    p = [0, 0.01, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.30, 0.40, 0.50]
    for i in p:
        X, y = load_boston(return_X_y=True)
        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=42)
        W, z = corrupt(X_tr, y_tr, outlier_ratio=i, random_state=42)

        reg = ENOLS(sample_size=42)
        reg.fit(W, z, random_state=42)
        ols, ts_reg, enols = reg.predict(X_ts, method='median')

        mse_ols = mean_squared_error(y_ts, ols)
        mse_ts_reg = mean_squared_error(y_ts, ts_reg)
        mse_enols = mean_squared_error(y_ts, enols)

        ensemble_ols.append(mse_enols), tsregressor.append(mse_ts_reg), ordinaryleastsquare.append(mse_ols)

    plt.plot(p, ensemble_ols, 'b', label="enols")
    plt.plot(p, tsregressor, 'r', label="tsr")
    plt.plot(p, ordinaryleastsquare, 'g', label="ols")
    plt.legend(loc="upper right")
    plt.xlabel('p', fontsize=18)
    plt.ylabel('mse', fontsize=16)
    plt.show()
