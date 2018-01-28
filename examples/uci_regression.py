#!/usr/bin/env python
"""
UCI regression
"""
import os
ROOT_DIR = os.environ['ROOT_DIR']
import sys
sys.path.append(ROOT_DIR)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import edward as ed
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from edward.inferences.ab_divergence import ABDivergence
from edward.models import Normal
from edward.util import Progbar
from observations import boston_housing
from sklearn.model_selection import KFold

# Load data:
dataset = 'boston_housing'
p_data = os.path.join(ROOT_DIR, 'data', 'raw', dataset)
data, info = boston_housing(p_data)

X = data[:, :-1]
y = np.expand_dims(data[:, -1], axis=-1)

# PARAMETERS
hidden = [50]
n_sample = 5
n_epoch = 500
l_alpha = np.array(list(np.arange(-1, 1.25, 0.25)) + [2])
l_beta = np.arange(-1, 1.25, 0.25)

outdir = os.path.join(ROOT_DIR, 'data', 'processed', dataset)
os.makedirs(outdir)
outname = os.path.join(outdir, 'H{}_N{}_S{}.h5'.format(hidden,
                                                       n_epoch,
                                                       n_sample))

column = ['outer_fold', 'inner_fold', 'alpha', 'beta', 'hidden',
          'n_epoch', 'n_sample', 'rmse', 'll']

# First fold - to
kf_outer = KFold(10, shuffle=True, random_state=np.random.seed(42))
l_df_outer = []
outer = 0
for train_outer_index, test_outer_index in kf_outer.split(X):
    X_train_outer = X[train_outer_index, :]
    y_train_outer = y[train_outer_index, :]

    X_test_outer = X[test_outer_index, :]
    y_test_outer = y[test_outer_index, :]

    # Normalize:
    std_X_train_outer = np.std(X_train_outer, 0)
    std_X_train_outer[std_X_train_outer == 0] = 1
    mean_X_train_outer = np.mean(X_train_outer, 0)
    X_train_outer = (X_train_outer - mean_X_train_outer) / std_X_train_outer
    X_test_outer = (X_test_outer - mean_X_train_outer) / std_X_train
    mean_y_train_outer = np.mean(y_train_outer)
    std_y_train_outer = np.std(y_train_outer)
    y_train_outer = (y_train_outer - mean_y_train_outer) / std_y_train_outer
    y_test_outer = (y_test_outer - mean_y_train_outer) / std_y_train_outer

    # 2nd Kfold validation to select the best alpha/beta
    kf_inner = KFold(4, shuffle=True, random_state=np.random.seed(42))
    l_df_inner = []
    inner = 0
    for train_inner_index, test_inner_index in kf_inner.split(X_train_outer):
        l_df_ab = []
        iter_ab = list(itertools.product(l_alpha, l_beta))
        for ab, (alpha, beta) in enumerate(iter_ab):
            print("OUTER: {}/{} - INNER: {}/{} - ab: {}/{}".format(outer + 1,
                                                                   kf_outer.n_splits,
                                                                   inner + 1,
                                                                   kf_inner.n_splits,
                                                                   ab + 1,
                                                                   len(iter_ab)))

            X_train_inner = X_train_outer[train_inner_index, :]
            y_train_inner = y_train_outer[train_inner_index, :]

            X_test_inner = X_test_outer[test_inner_index, :]
            y_test_inner = X_test_outer[test_inner_index, :]

            # Learning and test:
            rmse, ll = fit(X_train=X_train_inner, y_train=y_train_inner,
                           X_test=X_test_inner, y_test=y_test_inner,
                           alpha=alpha, beta=beta,
                           hidden=hidden, n_sample=n_sample, n_epoch=n_epoch)

            line = [outer, inner, alpha, beta, hidden, n_epoch,
                    n_sample, rmse, ll]
            line_df = pd.DataFrame([line], columns=column)
            line_df.to_hdf(outname, 'df')
            l_df_ab.append(line_df)

        inner += 1
        l_df_inner.append(pd.concat(l_df_ab))
        df_inner = pd.concat(l_df_inner)
        df_inner.to_hdf(outname, 'df')
    l_df_outer.append(df_inner)
    df_outer = pd.concat(l_df_outer)
    df_outer.to_hdf(outname, 'df', 'df')
    outer += 1
df = pd.concat(l_df_outer)


#############
### UTILS ###
#############
def neural_network(x, W, b):
    assert len(W.keys()) == len(b.keys())

    if len(W.keys()) == 0:
        print('here')
        h = tf.matmul(x, W[W.keys()[0]]) + W[W.keys()[1]]
    else:
        for i in range(len(W.keys())):
            if i == 0:
                h = tf.nn.relu(
                    tf.matmul(x, W["W_{}".format(i)]) + b["b_{}".format(i)])
            elif i == len(W.keys()) - 1:
                h = tf.matmul(h, W["W_{}".format(i)]) + b["b_{}".format(i)]
            else:
                h = tf.nn.relu(
                    tf.matmul(h, W["W_{}".format(i)]) + b["b_{}".format(i)])

    return h


def make_model(hidden, n_in=10, n_out=1):
    W = {}
    b = {}
    if len(hidden) == 0:
        W['W'] = Normal(loc=tf.zeros([n_in, n_out]),
                        scale=tf.ones([n_in, n_out]), name='W')
        b = Normal(loc=tf.zeros(n_out), scale=tf.ones(n_out), name="b")
    else:
        for h, n_neur in enumerate(hidden):

            if h == 0:
                W['W_{}'.format(h)] = Normal(loc=tf.zeros([n_in, n_neur]),
                                             scale=tf.ones([n_in, n_neur]), name='W_{}'.format(h))
                b['b_{}'.format(h)] = Normal(loc=tf.zeros(n_neur),
                                             scale=tf.ones(n_neur), name="b_{}".format(h))
            else:
                W['W_{}'.format(h)] = Normal(loc=tf.zeros([n_prev, n_neur]),
                                             scale=tf.ones([n_prev, n_neur]), name='W_{}'.format(h))
                b['b_{}'.format(h)] = Normal(loc=tf.zeros(n_neur),
                                             scale=tf.ones(n_neur), name="b_{}".format(h))
            n_prev = n_neur

        h += 1
        W['W_{}'.format(h)] = Normal(loc=tf.zeros([n_prev, n_out]),
                                     scale=tf.ones([n_prev, n_out]), name='W_{}'.format(h))
        b['b_{}'.format(h)] = Normal(loc=tf.zeros(n_out),
                                     scale=tf.ones(n_out), name="b_{}".format(h))

        X = tf.placeholder(tf.float32, [None, n_in], name="X")
        y = Normal(loc=neural_network(X, W, b), scale=tf.ones(1), name="y")

    return W, b, X, y


def make_inference(W, b, inference='AB_divergence', n_out=1):
    # INFERENCE
    qW = {}
    qb = {}
    with tf.name_scope("posterior"):
        for key in W.keys():
            with tf.name_scope("q{}".format(key)):
                qW["q{}".format(key)] = Normal(loc=tf.Variable(tf.random_normal(W[key].shape),
                                                               name="loc"),
                                               scale=tf.nn.softplus(
                    tf.Variable(tf.random_normal(W[key].shape),
                                name="scale")))
        for key in b.keys():
            with tf.name_scope("q{}".format(key)):
                qb["q{}".format(key)] = Normal(loc=tf.Variable(tf.random_normal(b[key].shape), name="loc"),
                                               scale=tf.nn.softplus(
                                                   tf.Variable(tf.random_normal(b[key].shape), name="scale")))

    inference_dic = {}
    for key in W.keys():
        inference_dic[W[key]] = qW["q{}".format(key)]
    for key in b.keys():
        inference_dic[b[key]] = qb["q{}".format(key)]

    y_ph = tf.placeholder(tf.float32, [None, 1], name='y_ph')

    if inference == 'AB_divergence':
        inference = ABDivergence(inference_dic, data={y: y_ph})

    return qW, qb, y_ph, inference, inference_dic


def make_batches(N_data, batch_size):
    return [slice(i, min(i + batch_size, N_data))
            for i in range(0, N_data, batch_size)]


def fit(X_train, y_train, X_test, y_test,
        alpha=1.0, beta=0.0, hidden=[50],
        n_sample=5, n_epoch=500, batch_size=32,
        n_in=10, n_out=1):
    # MODEL
    with tf.name_scope("model"):
        W, b, X, y = make_model(hidden=hidden, n_in=n_in, n_out=n_out)

    # INFERENCE
    with tf.name_scope("posterior"):
        qW, qb, y_ph, inference, inference_dic = make_inference(W, b)

    optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)

    inference.initialize(optimizer=optimizer, n_samples=n_sample,
                         alpha=alpha, beta=beta)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    batch_idxs = make_batches(X_train.shape[0], batch_size)

    for epoch in range(1, n_epoch + 1):
        print("Epoch: {0}".format(epoch))
        avg_loss = 0.0

        permutation = np.random.choice(
            range(X_train.shape[0]), X_train.shape[0], replace=False)

        pbar = Progbar(len(batch_idxs) - 1)
        for t, idxs in enumerate(batch_idxs):
            pbar.update(t)
            x_batch = X_train[permutation[idxs], :]
            y_batch = Y_train[permutation[idxs], :]

            info_dict = inference.update(feed_dict={X: x_batch,
                                                    y_ph: y_batch})
            avg_loss += info_dict['loss']

        y_post = ed.copy(y, inference_dic)

    rmse = np.sqrt(ed.evaluate('mean_squared_error',
                               data={X: X_test, y_post: y_test}))
    ll = np.sqrt(ed.evaluate('log_likelihood',
                               data={X: X_test, y_post: y_test}))

    return rmse, ll
