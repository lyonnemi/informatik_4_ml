import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random



def make_label(price_list):
    # input: price_list is a list of house prices
    # task: compute the mean of prices
    mean_price = sum(price_list) / len(price_list)
    # task: if price < mean -> label 1, else -> label 0
    # output: return the label list (same length as price_list)
    return [1 if p < mean_price else 0 for p in price_list]


def train_test_split(df, ratio):
    # input: df is the full dataframe, ratio like 0.8
    # task: take first ratio part as train, rest as test (no shuffle)
    train_size = int(len(df) * ratio)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    return train_df, test_df


def pick_xy(df, feature_names, label_name):
    x = df[feature_names].to_numpy(dtype=float)
    y = df[label_name].to_numpy(dtype=float)
    return x, y


def min_max_fit(x):
    # input: x is numpy array (n, d)
    # task: for each column, find min and max
    min_list = np.min(x, axis=0)
    max_list = np.max(x, axis=0)
    # output: return min_list, max_list (each length d)

    # n samples and d features
    return min_list, max_list


def min_max_transform(x, min_list, max_list):
    # input: x is numpy array (n, d), min_list and max_list from train data
    # task: scale each column to [0, 1] with (x - min) / (max - min)
    scaled = (x - min_list) / (max_list - min_list)
    # task: if max == min, set scaled value to 0
    scaled[:, max_list == min_list] = 0
    # output: return scaled x
    return scaled



def add_bias(x):
    n = x.shape[0]
    d = x.shape[1]
    out = np.zeros((n, d + 1), dtype=float)

    i = 0
    while i < n:
        out[i, 0] = 1.0
        j = 0
        while j < d:
            out[i, j + 1] = x[i, j]
            j = j + 1
        i = i + 1

    return out


def sigmoid(z):
    # input: z is a number or array
    # task: return 1 / (1 + exp(-z))
    # hint: for stability you can use the if z>=0 else form
    if np.isscalar(z):
        if z >= 0:
            return 1 / (1 + np.exp(-z))
        else:
            return np.exp(z) / (1 + np.exp(z))
    else:
        # vectorized version for arrays
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def log_cost(y, p):
    # input: y is 0/1 labels, p is predicted probabilities (0..1)
    # task: compute average cross-entropy loss:
    #       -mean( y*log(p) + (1-y)*log(1-p) )
    # hint: clip p to avoid log(0) using 1e-12
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def train_gd(x_train, y_train, x_test, y_test, lr, steps):
    # input: x_train (n,d), y_train (n,), lr, steps
    # task: initialize theta with zeros (length d)
    theta = np.zeros(x_train.shape[1], dtype=float)

    train_cost_list = []
    test_cost_list = []
    theta_history_list = []

    for step in range(steps):
        #   1) compute p_train for all training samples
        p_train = sigmoid(np.dot(x_train, theta))
        #   2) compute p_test for all test samples
        p_test = sigmoid(np.dot(x_test, theta))
        #   3) store train_cost and test_cost using log_cost
        train_cost = log_cost(y_train, p_train)
        test_cost = log_cost(y_test, p_test)
        train_cost_list.append(train_cost)
        test_cost_list.append(test_cost)
        #   4) compute gradient for each theta j:
        grad = np.zeros_like(theta)
        for j in range(theta.shape[0]):
            grad[j] = (1 / x_train.shape[0]) * np.sum((p_train - y_train) * x_train[:, j])
        #   5) update theta: theta = theta - lr * grad
        theta = theta - lr * grad

        #   6) store theta copy in theta history
        theta_history_list.append(theta.copy())

    #return final theta, train_cost_list, test_cost_list, theta_history_list
    return theta, train_cost_list, test_cost_list, theta_history_list



def predict_prob_one(x_row, t):
    z = float(np.dot(x_row, t))
    return sigmoid(z)


def predict_label(x, t):
    n = x.shape[0]
    out = []
    i = 0
    while i < n:
        p = predict_prob_one(x[i], t)
        if p >= 0.5:
            out.append(1)
        else:
            out.append(0)
        i = i + 1
    return np.array(out, dtype=float)


def accuracy(y_true, y_pred):
    n = len(y_true)
    ok = 0
    i = 0
    while i < n:
        if int(y_true[i]) == int(y_pred[i]):
            ok = ok + 1
        i = i + 1
    return ok / n


df = pd.read_csv("Real_estate.csv")

price = df["Y house price of unit area"].tolist()
lab = make_label(price)
df["label"] = lab

train_df, test_df = train_test_split(df, 0.8)

features = [
    "X1 transaction date",
    "X2 house age",
    "X3 distance to the nearest MRT station",
    "X4 number of convenience stores",
    "X5 latitude",
    "X6 longitude",
]

x_train, y_train = pick_xy(train_df, features, "label")
x_test, y_test = pick_xy(test_df, features, "label")

min_list, max_list = min_max_fit(x_train)
x_train = min_max_transform(x_train, min_list, max_list)
x_test = min_max_transform(x_test, min_list, max_list)

x_train = add_bias(x_train)
x_test = add_bias(x_test)

lr_list = [0.01, 0.1, 0.5]
steps = 1000

i = 0
while i < len(lr_list):
    lr = lr_list[i]

    t, train_cost, test_cost, t_hist = train_gd(x_train, y_train, x_test, y_test, lr, steps)

    plt.figure(figsize=(10, 4))
    plt.plot(train_cost, label="train")
    plt.plot(test_cost, label="test")
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.title("learning rate = " + str(lr))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    j = 0
    while j < len(t_hist[0]):
        one_line = []
        k = 0
        while k < len(t_hist):
            one_line.append(t_hist[k][j])
            k = k + 1
        plt.plot(one_line, label="theta " + str(j))
        j = j + 1
    plt.xlabel("iteration")
    plt.ylabel("theta value")
    plt.title("theta convergence, lr = " + str(lr))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    y_train_pred = predict_label(x_train, t)
    y_test_pred = predict_label(x_test, t)

    train_acc = accuracy(y_train, y_train_pred)
    test_acc = accuracy(y_test, y_test_pred)

    print("learning rate:", lr)
    print("training accuracy:", float(train_acc))
    print("test accuracy:", float(test_acc))
    print("")

    i = i + 1