import os
ROOT_DIR = os.environ['ROOT_DIR']
import sys
sys.path.append(ROOT_DIR)

import itertools
import math
import numpy as np
import os
import pandas as pd
import pickle
import time
import sys

import edward as ed
from edward.models import Normal
from edward.util import Progbar

import tensorflow as tf
# ed.set_seed(42)

# Helper functions:
def neural_network(X, W_0, b_0, W_1, b_1):
  """
  Creare a 1 layer neural network with relu activation
  """
  h = tf.nn.relu(tf.matmul(X, W_0) + b_0)
  out = tf.matmul(h, W_1) + b_1
  # h = tf.identity(tf.matmul(h, W_1) + b_1)
  return tf.reshape(out, [-1])


def make_batches(N_data, batch_size):
  """
  Prepare the batch indexes
  """
  return [slice(k, min(k + batch_size, N_data))
          for k in range(0, N_data, batch_size)]


def load_data(datapath):
  """
  Load a dataset
  """
  # We fix the random seed
  # TODO add test datapath in boston, yacht, wine, ...
  np.random.seed(1)

  # Number of splits:
  with open(os.path.join(datapath, 'n_splits.txt')) as f:
    n_splits = int(f.readlines()[0])

  # datapath = os.path.join('data', dataset)
  data = np.loadtxt(os.path.join(datapath, 'data.txt'))
  index_features = np.loadtxt(os.path.join(datapath, 'index_features.txt'),
                              dtype=int)
  index_target = np.loadtxt(os.path.join(datapath, 'index_target.txt'),
                            dtype=int)

  X = data[:, index_features.tolist()]
  y = data[:, index_target.tolist()]

  l_X_train = []
  l_y_train = []
  l_X_test = []
  l_y_test = []

  for split in range(n_splits + 1)[1:]:
    # We load the indexes of the training and test sets fro this spl
    index_train = np.loadtxt(os.path.join(datapath,
                                          "index_train_{}.txt".format(split)),
                             dtype=int)
    index_test = np.loadtxt(os.path.join(datapath,
                                         "index_test_{}.txt".format(split)),
                            dtype=int)

    # load training and test data
    X_train = X[index_train.tolist(), ]
    y_train = y[index_train.tolist()]
    X_test = X[index_test.tolist(), ]
    y_test = y[index_test.tolist()]

    # We normalize the features
    std_X_train = np.std(X_train, 0)
    std_X_train[std_X_train == 0] = 1
    mean_X_train = np.mean(X_train, 0)
    X_train = (X_train - mean_X_train) / std_X_train
    X_test = (X_test - mean_X_train) / std_X_train
    mean_y_train = np.mean(y_train)
    std_y_train = np.std(y_train)
    y_train = (y_train - mean_y_train) / std_y_train
    y_test = (y_test - mean_y_train) / std_y_train

    y_train = np.array(y_train, ndmin=2).reshape((-1, 1))
    y_test = np.array(y_test, ndmin=2).reshape((-1, 1))

    l_X_train.append(X_train.astype(np.float32))
    l_y_train.append(y_train.astype(np.float32))
    l_X_test.append(X_test.astype(np.float32))
    l_y_test.append(y_test.astype(np.float32))

  return l_X_train, l_y_train, l_X_test, l_y_test


# PARAMETERS:
dataset = 'yacht'
p_dataset = os.path.join(ROOT_DIR, 'data', 'processed', dataset)
logs_path = os.path.join(ROOT_DIR, 'models', dataset, 'kl')
n_samples = 100     # number of samples used to estimate the Renyi ELBO
n_neurons = 50
batch_size = 32
n_epoch = 500

# OUTPUT
col = ['dataset', 'N', 'D', 'mean_rmse', 'var_rmse', 'mean_ll', 'var_ll']
output_name = '{}_KL_E{}_B{}_K{}.h5'.format(dataset,
                                            n_epoch,
                                            batch_size,
                                            n_samples)
out = pd.DataFrame(columns=col)

# DATA
X_train, y_train, X_test, y_test = load_data(p_dataset)
n_splits = len(X_train)

sess = ed.get_session()
print("--- Dataset: {}".format(dataset))
print("------ Datapoints: {}".format(X_train[0].shape[0] + X_test[0].shape[0]))
print("------ Features: {}".format(X_train[0].shape[1]))

test_ll = np.zeros(n_splits)
test_rmse = np.zeros(n_splits)
for i in range(n_splits):
  print("Split {}/{}".format(i + 1, n_splits))
  N, D = X_train[i].shape   # number of training points and features

  # MODEL
  if i == 0:
    with tf.name_scope("model"):
      print("model")
      W_0 = Normal(loc=tf.zeros([D, n_neurons]),
                   scale=tf.ones([D, n_neurons]), name="W_0")
      W_1 = Normal(loc=tf.zeros([n_neurons, 1]),
                   scale=tf.ones([n_neurons, 1]), name="W_1")
      b_0 = Normal(loc=tf.zeros(n_neurons),
                   scale=tf.ones(n_neurons), name="b_0")
      b_1 = Normal(loc=tf.zeros(1), scale=tf.ones(1), name="b_1")

      X = tf.placeholder(tf.float32, [None, D], name="X")
      y_ph = tf.placeholder(tf.float32, [None])
      y = Normal(loc=neural_network(X, W_0=W_0, b_0=b_0, W_1=W_1, b_1=b_1),
                 scale=0.1 * tf.ones(1), name="y")

    # INFERENCE
    with tf.name_scope("posterior"):
      print("posterior")
      with tf.name_scope("qW_0"):
        qW_0 = Normal(loc=tf.Variable(tf.random_normal([D, n_neurons]),
                                      name="loc"),
                      scale=tf.nn.softplus(
            tf.Variable(tf.random_normal([D, n_neurons]),
                        name="scale")))
      with tf.name_scope("qW_1"):
        qW_1 = Normal(loc=tf.Variable(tf.random_normal([n_neurons, 1]),
                                      name="loc"),
                      scale=tf.nn.softplus(
            tf.Variable(tf.random_normal([n_neurons, 1]),
                        name="scale")))
      with tf.name_scope("qb_0"):
        qb_0 = Normal(loc=tf.Variable(tf.random_normal([n_neurons]),
                                      name="loc"),
                      scale=tf.nn.softplus(
                          tf.Variable(tf.random_normal([n_neurons]),
                                      name="scale")))
      with tf.name_scope("qb_1"):
        qb_1 = Normal(loc=tf.Variable(tf.random_normal([1]), name="loc"),
                      scale=tf.nn.softplus(
                          tf.Variable(tf.random_normal([1]), name="scale")))

    with tf.name_scope("inference"):
      print("inference")
      inference = ed.KLqp({W_0: qW_0, b_0: qb_0, W_1: qW_1, b_1: qb_1},
                          data={y: y_ph})
  #   data={X: X_train[i],
  #         y_ph: np.ravel(y_train[i])})
    with tf.name_scope("optimizer"):
      print("optimizer")
      optimizer = tf.train.AdamOptimizer(learning_rate=0.01,
                                         beta1=0.9,
                                         beta2=0.999,
                                         epsilon=10e-8)

      b_index = make_batches(N, batch_size)
      inference.initialize(optimizer=optimizer,
                           n_samples=n_samples,
                           scale={y: batch_size / len(b_index)})
  else:
    b_index = make_batches(N, batch_size)

  print("initializer")
  tf.global_variables_initializer().run()

  if i == 0:
    print("writer")
    # create log writer object
    writer = tf.summary.FileWriter(logs_path,
                                   graph=tf.get_default_graph())

  print("here")
  pbar = Progbar(n_epoch)
  for epoch in range(n_epoch):
    pbar.update(epoch + 1)
    avg_loss = 0.0
    for j, slce in enumerate(b_index):
      x_t = X_train[i][slce]
      y_t = y_train[i][slce]
      info_dict = inference.update(feed_dict={X: x_t,
                                              y_ph: np.ravel(y_t)})
      avg_loss += info_dict['loss']
      # write log
    #   writer.add_summary(summary, epoch * len(b_index) + j)

    # Print a lower bound to the average marginal likelihood for an
    # image.
    avg_loss = avg_loss / len(b_index)
    avg_loss = avg_loss / batch_size
  print("log p(x) >= {:0.3f}".format(avg_loss))

  # build posterior predictive on test data
  N_test, D_test = X_test[i].shape   # number of features
  y_post = ed.copy(y, {W_0: qW_0, b_0: qb_0, W_1: qW_1, b_1: qb_1})

  # Compute RMSE and N-LL
  tmp_nll = ed.evaluate('log_likelihood',
                        data={X: X_test[i],
                              y_post: np.ravel(y_test[i])})
  print("Split {} - N-LL= {}".format(i, tmp_nll))
  test_ll[i] = tmp_nll

  tmp_rmse = np.sqrt(
      ed.evaluate('mean_squared_error',
                  data={X: X_test[i],
                        y_post: np.ravel(y_test[i])}))
  print("Split {} - RMSE = {}".format(i, tmp_rmse))
  test_rmse[i] = tmp_rmse

mean_rmse = np.mean(np.asarray(test_rmse))
var_rmse = np.var(np.asarray(test_rmse))
mean_ll = np.mean(np.asarray(test_ll))
var_ll = np.var(np.asarray(test_ll))

pd_info = [dataset, N + N_test, D, mean_rmse, var_rmse, mean_ll, var_ll]
out = out.append(pd.DataFrame([pd_info], columns=col))

# print("--- Dataset: {}".format(dataset))
print(pd_info)
# print('log_likelihood - Mean= {} - Variance= {}'.format(
#     mean_ll, var_ll))
# print('log_rmse - Mean= {} - Variance= {}'.format(
#     mean_rmse, var_rmse))

out.to_hdf(output_name, 'table')
