import ssl

import numpy as np
import torch
import torch.optim as optim
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable

ssl._create_default_https_context = ssl._create_unverified_context


class LogisticRegression:
    def __init__(self):
        pass

    def fit(self, X, y, X_ts, y_ts, lr=0.01, momentum=0.9, niter=100):
        """
        Train a multiclass logistic regression model on the given training set.

        Parameters
        ----------
        X: training examples, represented as an input array of shape (n_sample,
           n_features).
        y: labels of training examples, represented as an array of shape
           (n_sample,) containing the classes for the input examples
        lr: learning rate for gradient descent
        niter: number of gradient descent updates
        momentum: the momentum constant (see assignment task sheet for an explanation)

        Returns
        -------
        self: fitted model
        """
        self.classes_ = np.unique(y)

        n_features = X.shape[1]
        n_classes = len(self.classes_) + 1

        model = LogisticRegression().linear_model(n_features, n_classes)
        loss = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        for i in range(niter):
            cost = 0
            num_batches = X.shape[0] // niter
            for batch in range(num_batches):
                start, end = batch * niter, (batch + 1) * niter
                cost += LogisticRegression().train_model(model, loss, optimizer, X[start:end], y[start:end])
            y_pred = LogisticRegression().predict(model, X_ts)
            print(f"Epoch {i + 1}, cost = {cost / num_batches}, training_accuracy: "
                  f"{100 * LogisticRegression().calculate_accuracy(model, X, y)}, "
                  f"test_accuracy = {100 * np.mean(y_pred == y_ts)}")
        print(LogisticRegression().predict_proba(model, X_ts))
        return self

    def calculate_accuracy(self, model, X, y):
        max_indices = torch.max(model(X), 1)[1]
        return (max_indices == y).sum().item() / max_indices.size(0)

    def train_model(self, model, loss, optimizer, X, y):
        model.train()
        x = Variable(X, requires_grad=False)
        y = Variable(y, requires_grad=False)

        optimizer.zero_grad()

        output = loss.forward(model.forward(x), y)
        output.backward()

        optimizer.step()

        return output.item()

    def predict_proba(self, model, X):
        """
        Predict the class distributions for given input examples.

        Parameters
        ----------
        X: input examples, represented as an input array of shape (n_sample,
           n_features).

        Returns
        -------
        y: predicted class labels, represented as an array of shape (n_sample,
           n_classes)
        """
        model.eval()
        output = model.forward(Variable(X, requires_grad=False))
        prob = torch.nn.functional.softmax(output)
        return prob

    def linear_model(self, input_dim, output_dim):
        model = torch.nn.Sequential()
        model.add_module("linear", torch.nn.Linear(input_dim, output_dim, bias=True))
        return model

    def predict(self, model, X):
        """
        Predict the classes for given input examples.

        Parameters
        ----------
        X: input examples, represented as an input array of shape (n_sample,
           n_features).

        Returns
        -------
        y: predicted class labels, represented as an array of shape (n_sample,)
        """
        model.eval()
        output = model.forward(Variable(X, requires_grad=False))
        return output.data.numpy().argmax(axis=1)


if __name__ == '__main__':
    X, y = fetch_covtype(return_X_y=True)
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=42)

    sc = StandardScaler()
    X_tr = torch.from_numpy(sc.fit_transform(X_tr)).float()
    X_ts = torch.from_numpy(sc.fit_transform(X_ts)).float()
    y_tr = torch.from_numpy(y_tr).long()

    logistic = LogisticRegression()
    logistic.fit(X_tr, y_tr, X_ts, y_ts)

