import tqdm
import time
import json
import pandas as pd
import numpy as np
import copy
import random
from load_dataset import load, generate_random_dataset
from classifier import LogisticRegression, NeuralNetwork
from utils import *
from metrics import *  # include fairness and corresponding derivatives
from sklearn.preprocessing import StandardScaler
from operator import itemgetter
from torch.autograd import grad
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dup', type=int, default=1)
parser.add_argument('--per', type=int, default=5)

args = parser.parse_args()

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

dataset = 'german'
per = args.per / 100
repeat = 50

X_train, X_test, y_train, y_test = load(dataset)
make_duplicates_pd = lambda x, d: pd.concat([x] * d, axis=0).reset_index(drop=True)
make_duplicates_np = lambda x, d: np.concatenate([x] * d, axis=0)
# X_train = make_duplicates(X_train, duplicates)
# X_test = make_duplicates(X_test, duplicates)
# y_train = make_duplicates(y_train, duplicates)
# y_test = make_duplicates(y_test, duplicates)

X_train_orig = copy.deepcopy(X_train)
X_test_orig = copy.deepcopy(X_test)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train_orig_single = X_train_orig.copy()
X_test_orig_single = X_test_orig.copy()
X_train_single = X_train.copy()
X_test_single = X_test.copy()
y_train_single = y_train.copy()
y_test_single = y_test.copy()

clf = LogisticRegression(input_size=X_train.shape[-1])
# clf = NeuralNetwork(input_size=X_train.shape[-1])
num_params = len(convert_grad_to_ndarray(list(clf.parameters())))
loss_func = logistic_loss_torch


def ground_truth_influence(X_train, y_train, X_test, X_test_orig, y_test):
    clf.fit(X_train, y_train, verbose=True)
    y_pred = clf.predict_proba(X_test)
    spd_0 = computeFairness(y_pred, X_test_orig, y_test, 0)

    delta_spd = []
    for i in range(len(X_train)):
        X_removed = np.delete(X_train, i, 0)
        y_removed = y_train.drop(index=i, inplace=False)
        clf.fit(X_removed, y_removed)
        y_pred = clf.predict_proba(X_test)
        delta_spd_i = computeFairness(y_pred, X_test_orig, y_test, 0) - spd_0
        delta_spd.append(delta_spd_i)

    return delta_spd


def computeAccuracy(y_true, y_pred):
    return np.sum((y_pred > 0.5) == y_true) / len(y_pred)


def del_L_del_theta_i(model, x, y_true, retain_graph=False):
    loss = loss_func(model, x, y_true)
    w = [p for p in model.parameters() if p.requires_grad]
    return grad(loss, w, create_graph=True, retain_graph=retain_graph)


def del_f_del_theta_i(model, x, retain_graph=False):
    w = [p for p in model.parameters() if p.requires_grad]
    return grad(model(torch.FloatTensor(x)), w, retain_graph=retain_graph)


def hvp(y, w, v):
    """ Multiply the Hessians of y and w by v."""
    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(convert_grad_to_tensor(first_grads), v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads


def hessian_one_point(model, x, y):
    x, y = torch.FloatTensor(x), torch.FloatTensor([y])
    loss = loss_func(model, x, y)
    params = [p for p in model.parameters() if p.requires_grad]
    first_grads = convert_grad_to_tensor(grad(loss, params, retain_graph=True, create_graph=True))
    hv = np.zeros((len(first_grads), len(first_grads)))
    for i in range(len(first_grads)):
        hv[i, :] = convert_grad_to_ndarray(grad(first_grads[i], params, create_graph=True)).ravel()
    return hv


# Compute multiplication of inverse hessian matrix and vector v
def s_test(model, xs, ys, v, hinv=None, damp=0.01, scale=25.0, r=-1, batch_size=-1, recursive=False, verbose=False):
    """ Arguments:
        xs: list of data points
        ys: list of true labels corresponding to data points in xs
        damp: dampening factor
        scale: scaling factor
        r: number of iterations aka recursion depth
            should be enough so that the value stabilises.
        batch_size: number of instances in each batch in recursive approximation
        recursive: determine whether to recursively approximate hinv_v"""
    xs, ys = torch.FloatTensor(xs.copy()), torch.FloatTensor(ys.copy())
    n = len(xs)
    if recursive:
        hinv_v = copy.deepcopy(v)
        if verbose:
            print('Computing s_test...')
            tbar = tqdm.tqdm(total=r)
        if batch_size == -1:  # default
            batch_size = 10
        if r == -1:
            r = n // batch_size + 1
        sample = np.random.choice(range(n), r * batch_size, replace=True)
        for i in range(r):
            sample_idx = sample[i * batch_size:(i + 1) * batch_size]
            x, y = xs[sample_idx], ys[sample_idx]
            loss = loss_func(model, x, y)
            params = [p for p in model.parameters() if p.requires_grad]
            hv = convert_grad_to_ndarray(hvp(loss, params, torch.FloatTensor(hinv_v)))
            # Recursively caclulate h_estimate
            hinv_v = v + (1 - damp) * hinv_v - hv / scale
            if verbose:
                tbar.update(1)
    else:
        if hinv is None:
            hinv = np.linalg.pinv(np.sum(hessian_all_points, axis=0))
        scale = 1.0
        hinv_v = np.matmul(hinv, v)

    return hinv_v / scale


clf = LogisticRegression(input_size=X_train.shape[-1])
# clf = NeuralNetwork(input_size=X_train.shape[-1])
clf.fit(X_train, y_train)

y_pred_test = clf.predict_proba(X_test)
y_pred_train = clf.predict_proba(X_train)

spd_0 = computeFairness(y_pred_test, X_test_orig, y_test, 0, dataset)
print("Initial statistical parity: ", spd_0)

tpr_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 1, dataset)
print("Initial TPR parity: ", tpr_parity_0)

predictive_parity_0 = computeFairness(y_pred_test, X_test_orig, y_test, 2, dataset)
print("Initial predictive parity: ", predictive_parity_0)

loss_0 = logistic_loss(y_test, y_pred_test)
print("Initial loss: ", loss_0)

accuracy_0 = computeAccuracy(y_test, y_pred_test)
print("Initial accuracy: ", accuracy_0)

hessian_all_points = []
print('Computing Hessian')
tbar = tqdm.tqdm(total=len(X_train))
total_time = 0
for i in range(len(X_train)):
    t0 = time.time()
    hessian_all_points.append(hessian_one_point(clf, X_train[i], y_train[i]) / len(X_train))
    total_time += time.time() - t0
    tbar.update(1)

hessian_all_points = np.array(hessian_all_points)
hessian_all_points_single = hessian_all_points.copy()

del_L_del_theta = []
for i in range(int(len(X_train))):
    gradient = convert_grad_to_ndarray(del_L_del_theta_i(clf, X_train[i], int(y_train[i])))
    while np.sum(np.isnan(gradient)) > 0:
        gradient = convert_grad_to_ndarray(del_L_del_theta_i(clf, X_train[i], int(y_train[i])))
    del_L_del_theta.append(gradient)
del_L_del_theta = np.array(del_L_del_theta)
del_L_del_theta_single = np.array(del_L_del_theta)

metric = 0
if metric == 0:
    v1 = del_spd_del_theta(clf, X_test_orig, X_test, dataset)
elif metric == 1:
    v1 = del_tpr_parity_del_theta(clf, X_test_orig, X_test, y_test, dataset)
elif metric == 2:
    v1 = del_predictive_parity_del_theta(clf, X_test_orig, X_test, y_test, dataset)

hinv = np.linalg.pinv(np.sum(hessian_all_points, axis=0))
hinv_v = s_test(clf, X_train, y_train, v1, hinv=hinv, verbose=False)


def first_order_influence(del_L_del_theta, hinv_v, n):
    infs = []
    for i in range(n):
        inf = -np.dot(del_L_del_theta[i].transpose(), hinv_v)
        inf *= -1 / n
        infs.append(inf)
    return infs


def second_order_influence(model, X_train, y_train, U, del_L_del_theta, r=-1, verbose=False):
    u = len(U)
    s = len(X_train)
    p = u / s
    c1 = (1 - 2 * p) / (s * (1 - p) ** 2)
    c2 = 1 / ((s * (1 - p)) ** 2)
    num_params = len(del_L_del_theta[0])
    del_L_del_theta_sum = np.sum([del_L_del_theta[i] for i in U], axis=0)
    hinv_del_L_del_theta = s_test(model, X_train, y_train, del_L_del_theta_sum, hinv=hinv)
    hessian_U_hinv_del_L_del_theta = np.zeros((num_params,))
    for i in range(u):
        idx = U[i]
        x, y = torch.FloatTensor(X_train[idx]), torch.FloatTensor([y_train[idx]])
        loss = loss_func(model, x, y)
        params = [p for p in model.parameters() if p.requires_grad]
        hessian_U_hinv_del_L_del_theta += convert_grad_to_ndarray(
            hvp(loss, params, torch.FloatTensor(hinv_del_L_del_theta)))

    term1 = c1 * hinv_del_L_del_theta
    term2 = c2 * s_test(model, X_train, y_train, hessian_U_hinv_del_L_del_theta, hinv=hinv)
    sum_term = term1 + term2
    return sum_term


def first_order_group_influence(U, del_L_del_theta):
    infs = []
    return 1 / len(X_train) * np.sum(np.dot(del_L_del_theta[U, :], hinv), axis=0)


def second_order_group_influence(U, del_L_del_theta):
    u = len(U)
    s = len(X_train)
    p = u / s
    c1 = (1 - 2 * p) / (s * (1 - p) ** 2)
    c2 = 1 / ((s * (1 - p)) ** 2)
    num_params = len(del_L_del_theta[0])
    del_L_del_theta_sum = np.sum(del_L_del_theta[U, :], axis=0)
    hinv_del_L_del_theta = np.matmul(hinv, del_L_del_theta_sum)
    hessian_U_hinv_del_L_del_theta = np.sum(np.matmul(hessian_all_points[U, :], hinv_del_L_del_theta), axis=0)
    term1 = c1 * hinv_del_L_del_theta
    term2 = c2 * np.matmul(hinv, hessian_U_hinv_del_L_del_theta)
    sum_term = (term1 + term2 * len(X_train))
    return sum_term


def get_subset(explanation):
    subset = X_train_orig.copy()
    for predicate in explanation:
        attr = predicate.split("=")[0].strip(' ')
        val = int(predicate.split("=")[1].strip(' '))
        subset = subset[subset[attr] == val]
    return subset.index


infs_1 = first_order_influence(del_L_del_theta, hinv_v, len(X_train))

dups = []
lr = 3
for dup in [50, 100, 200, 400, 800, 1600]:
    X_train_orig = make_duplicates_pd(X_train_orig_single, dup).copy()
    X_test_orig = make_duplicates_pd(X_test_orig_single, dup).copy()
    X_train = make_duplicates_np(X_train_single, dup).copy()
    X_test = make_duplicates_np(X_test_single, dup).copy()
    y_train = make_duplicates_pd(y_train_single, dup).copy()
    y_test = make_duplicates_pd(y_test_single, dup).copy()
    hessian_all_points = make_duplicates_np(hessian_all_points_single, dup).copy()
    del_L_del_theta = make_duplicates_np(del_L_del_theta_single, dup).copy()
    time_first = []
    time_second = []
    time_gt = []
    time_gd = []
    for r in range(repeat):
        sample_idx = np.random.choice(np.arange(len(X_train)), size=int(per / 100 * len(X_train)), replace=False).copy()
        idx = X_train_orig.iloc[sample_idx, :].index
        np.sum(np.dot(del_L_del_theta[U, :], hinv), axis=0)
        
        t0 = time.time()
        params_f_1 = first_order_group_influence(idx, del_L_del_theta)
        del_f_1 = np.dot(v1.transpose(), params_f_1)
        time_first.append(time.time() - t0)

        # Second-order subset influence
        t0 = time.time()
        params_f_2 = second_order_group_influence(idx, del_L_del_theta)
        del_f_2 = np.dot(v1.transpose(), params_f_2)
        time_second.append(time.time() - t0)

        # Ground truth subset influence (Retraining)
        clf.fit(np.array(X_train), np.array(y_train))
        t0 = time.time()
        X = np.delete(X_train, idx, 0)
        y = y_train.drop(index=idx, inplace=False)
        inf_gt = 0
        clf.fit(np.array(X), np.array(y))
        y_pred = clf.predict_proba(np.array(X_test))
        inf_gt = computeFairness(y_pred, X_test_orig, y_test, 0, dataset) - spd_0
        time_gt.append(time.time() - t0)

        clf.fit(np.array(X_train), np.array(y_train))
        optimizer = torch.optim.SGD(clf.parameters(), lr=lr)
        optimizer.zero_grad()
        clf.train()
        t0 = time.time()
        clf.partial_fit(X, y, learning_rate=lr)
        y_pred = clf.predict_proba(np.array(X_test))
        inf_gd = computeFairness(y_pred, X_test_orig, y_test, 0, dataset) - spd_0
        time_gd.append(time.time() - t0)

    dups.append([time_first, time_second, time_gt, time_gd])

np.save('dups.npy', np.array(dups))
