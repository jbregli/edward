"""
# Supervised Regression with outliers

In supervised learning, the task is to infer hidden structure from labeled data,
comprised of training examples $\{(x_n, y_n)\}$.
To demonstrate the robustness of the AB-divergence to outliers, the dataset used
has a percentage of corrupted values. ($5%$ in our experiments)

We demonstrate with an example in Edward.

Note:
This example is greatly inspired from the code available
http://edwardlib.org/tutorials/supervised-regression
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from edward.models import Normal
from edward.inferences.ab_divergence import ABDivergence

plt.style.use('ggplot')

np.random.seed(42)
ed.set_seed(42)

# EXPERIMENT PARAMETERS
N = 1000  # number of data points
D = 2  # number of features
P_OUTLIERS = 0.05
ALPHA = 1.0
BETA = 0.0
N_SAMPLES = 5
N_ITER = 7000
N_EXP = 50

# UTILS


def build_toy_dataset(N, w, p=1):
  """
  Construct a toy dataset for regression with outliers.
  'p' percent of the data follow:
    $y = w*x + \mcalN(0, 0.01)$ with $x = -2 + \mcalN(0, 2)$.
  '1-p' percent of the data follow
    $y = 10 + w*x + \mcalN(0, 0.01)$ with $x = \mcalN(0, 0.2)$.

  Args:
  -----
  N: int, number of points in te dataset.
  w: np.array, leading coefficient
  p: float (optional p=1.0), 1-p is the percentage of outliers.
  """
  N_out = int(np.floor(N * (1 - p)))
  N_cor = N - N_out

  # Correct points:
  D = len(w)
  x = -2 + np.random.normal(0.0, 2.0, size=(N_cor, D))
  y = np.dot(x, w) + np.random.normal(0.0, 0.01, size=N_cor)

  # Outliers:
  if N_out > 0:
    x_out = np.random.normal(0.0, 0.2, size=(N_out, D))
    y_out = 10 + np.dot(x_out, w) + np.random.normal(0.0, 0.01, size=N_out)

    x = np.vstack([x, x_out])

  return x, y


def visualise(X_data, y_data, w, b, n_samples=10, save=None):
  """
  Visualize the fit of an estimation.
  Plot the regression obtained using the mean of the leading
  y = np.hstack([y, y_out])
  coefficient and bias. Also displays the variance in the estimate.

  Args:
  -----
  X_data: np.array, values of the inputs.
  y_data: np.array, values of the outputs.
  w: abc.distribution, distribution of the leading coefficient.
  w: abc.distribution, distribution of the bias.
  n_samples: int (optional, 10), number of sample to evaluate the mean
  save: None or str
    Name to save the figure.
  """
  # fig = plt.figure(frameon=False)
  # ax = plt.Axes(fig, [0., 0., 1., 1.])

  w_samples = w.sample(n_samples)[:, 0].eval()
  b_samples = b.sample(n_samples).eval()
  plt.scatter(X_data[:, 0], y_data)
  plt.ylim([-10, 10])
  inputs = np.linspace(-8, 8, num=400)
  outputs = inputs * np.mean(w_samples) + np.mean(b_samples)
  errors = inputs * np.var(w_samples) + np.var(b_samples)

  plt.plot(inputs, outputs, color='g')
  plt.fill_between(inputs, outputs - errors, outputs + errors,
                   color='g', alpha=0.5)
  if save is not None:
    plt.savefig(save, format='png', frameon=False)


# DATA:
w_true = np.random.randn(D) * 0.5
X_train, y_train = build_toy_dataset(N, w_true, p=1 - P_OUTLIERS)
X_test, y_test = build_toy_dataset(N, w_true, p=1 - P_OUTLIERS)

# MODEL
X = tf.placeholder(tf.float32, [N, D])
w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y = Normal(loc=ed.dot(X, w) + b, scale=tf.ones(N))

# Inference
qw = Normal(loc=tf.Variable(tf.random_normal([D])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
qb = Normal(loc=tf.Variable(tf.random_normal([1])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

l_mae = []
l_mse = []
for exp in range(N_EXP):
  inference = ABDivergence({w: qw, b: qb}, data={X: X_train, y: y_train})
  optimizer = tf.train.AdamOptimizer(0.001)
  inference.run(n_samples=N_SAMPLES, n_iter=N_ITER,
                alpha=ALPHA, beta=BETA, optimizer=optimizer)

  # Criticism
  y_post = ed.copy(y, {w: qw, b: qb})
  mse = ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test})
  l_mse.append(mse)
  print("Mean squared error on test data: {}".format(mse))

  mae = ed.evaluate('mean_absolute_error', data={X: X_test, y_post: y_test})
  l_mae.append(mae)
  print("Mean absolute error on test data: {}".format(mae))

  # Visualize samples from the posterior.
  name_fig = os.path.join('..', 'models', 'regression_outliers', 'plots',
                          'a{}_b{}_{}.png'.format(ALPHA, BETA, exp))
  visualise(X_test, y_test, qw, qb, n_samples=100, save=name_fig)

l_alpha = [ALPHA] * N_EXP
l_beta = [BETA] * N_EXP
l_niter = [N_ITER] * N_EXP
l_nsamples = [N_SAMPLES] * N_EXP
df = pd.DataFrame({'alpha': l_alpha, 'beta': l_beta,
                   'n_iter': l_niter, 'n_samples': l_nsamples,
                   'mse': l_mse, 'mae': l_mae})
df.to_json(os.path.join('..', 'models', 'regression_outliers',
                        'a{}_b{}.json'.format(ALPHA, BETA)))
