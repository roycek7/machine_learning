import ssl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ssl._create_default_https_context = ssl._create_unverified_context


def evaluate(reg, X_tr, y_tr, X_ts, y_ts, N=1, seed=42):
    if seed:
        np.random.seed(seed)
    mse = []
    for i in range(N):
        reg.fit(X_tr, y_tr)
        r2_tr = reg.score(X_tr, y_tr)
        r2_ts = r2_score(y_ts, reg.predict(X_ts))
        mse_tr = mean_squared_error(y_tr, reg.predict(X_tr))
        mse_ts = mean_squared_error(y_ts, reg.predict(X_ts))
        mse.append([r2_tr, r2_ts, mse_tr, mse_ts])
    return pd.DataFrame(mse, columns=['R2 (train)', 'R2 (test)', 'MSE (train)', 'MSE (test)'])


def ensemble_random_forest(X, X_train, y_train, X_test, california, features):
    rf = RandomForestRegressor(n_estimators=100, max_features=features)
    rf.fit(X_train, y_train)

    perf_dt = evaluate(rf, *california, N=1)

    predictions = np.array([tree.predict(X_test) for tree in rf.estimators_])
    correlation = []
    count = 0
    for i in range(0, 100):
        correlation.append([pearsonr(predictions[i], predictions[j])[0] for j in range(0, 100)])
        count += sum(correlation[i])

    return count / len(correlation), perf_dt


def max_features():
    data = fetch_california_housing()
    X, y = data['data'], data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    california = X_train, y_train, X_test, y_test

    print(f'd: {len(X[0])}')
    print('Default:Auto m=n_features, i.e 8')

    pearson, training_error, test_error = [], [], []
    for i in range(len(X[0])):
        pearson_correlation_average, perf_dt = ensemble_random_forest(X, X_train, y_train, X_test, california, i + 1)
        pearson.append(pearson_correlation_average), training_error.append(perf_dt.iloc[0]["R2 (train)"]),
        test_error.append(perf_dt.iloc[0]["R2 (test)"])
        print(f'pearson_correlation average, using m = {i + 1}: {pearson_correlation_average}, \nperf_dt: {perf_dt}')
        print('----------------------------------------')

    plt.plot(training_error, 'b', label="Training")
    plt.plot(test_error, 'r', label="Test")
    plt.legend(loc="upper right")
    plt.xlabel('m', fontsize=18)
    plt.ylabel('mse', fontsize=16)
    plt.show()

    plt.plot(pearson, 'g', label="Average Correlation")
    plt.legend(loc="upper left")
    plt.xlabel('m', fontsize=18)
    plt.ylabel('pearson correlation', fontsize=16)
    plt.show()


max_features()
