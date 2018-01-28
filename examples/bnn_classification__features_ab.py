ROOT_DIR = os.environ['ROOT_DIR']
import os
import sys
sys.path.append(ROOT_DIR)

import itertools
import numpy as np
import pandas as pd

import edward as ed
from edward.inferences.ab_divergence import ABDivergence
from edward.models import Categorical, Normal
from edward.util import Progbar

from tensorflow.examples.tutorials.mnist import input_data

import copy

from scipy.misc import imsave

import tensorflow as tf
ed.set_seed(42)


# Helper functions:
def neural_network(x, W_0, b_0):
    """
    Create a 2 layers neural network with relu activation
    """
    out = tf.matmul(x, W_0) + b_0
    return out


def next_batch_outliers(dataset, batch_size, p_out, one_hot=False):
    """
    Prepare the batches
    """
    x, y = dataset.next_batch(batch_size)

    idx = np.random.binomial(1, p=p_out, size=batch_size)

    y_new = copy.deepcopy(y)
    y_new[np.where(idx)] = np.array([np.random.choice(np.delete(np.arange(10), y_old))
                                     for y_old in y[np.where(idx)]])

    if one_hot:
        y_new = tf.one_hot(y_new, 10)

    return x, y_new


# DATA
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'mnist')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# DATA. MNIST batches are fed at training time.
mnist = input_data.read_data_sets(DATA_DIR)

# PARAMETERS:
# , -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
l_alpha = [0.9] # , 0.5]
# alpha values for renyi divergence
# l_beta = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
l_beta = [-0.2] # , 0.0, 0.5]

# beta = 0.4      # alpha values for renyi divergence
n_samples = 5    # number of samples used to estimate the Renyi ELBO
batch_size = 32
logs_path = os.path.join(ROOT_DIR, 'models', 'mnist', 'ab')

# batch_size = 128   # number of images in a minibatch.
D = 784   # number of features.
K = 10    # number of classes.
n_iter = 1000000
p_outliers = 0.1

starter_learning_rate = 0.001
decay_ratio = 0.75
decay_step_size = 50000

# OUTPUT
col = ['n_iter', 'batch_size', 'n_samples',
       'p_outliers',
       'alpha', 'beta', 'accuracy']
output_name = 'mnist_AB_E{}_B{}_K{}.h5'.format(n_iter,
                                               batch_size,
                                               n_samples)
out = pd.DataFrame(columns=col)

sess = ed.get_session()
i = 0
for alpha, beta in itertools.product(reversed(l_alpha), reversed(l_beta)):
    print('alpha= {} - beta= {}'.format(alpha, beta))
    # MODEL
    if i == 0:
        with tf.name_scope("model"):
            W_0 = Normal(loc=tf.zeros([D, K]),
                         scale=tf.ones([D, K]),
                         name="W_0")
            b_0 = Normal(loc=tf.zeros(K),
                         scale=tf.ones(K),
                         name="b_0")

            X = tf.placeholder(tf.float32, [None, D], name="X")
            y = Categorical(neural_network(X, W_0, b_0), name="y")

        # INFERENCE
        with tf.name_scope("posterior"):
            with tf.name_scope("qW_0"):
                qW_0 = Normal(loc=tf.Variable(tf.random_normal([D, K]),
                                              name="loc"),
                              scale=tf.nn.softplus(
                              tf.Variable(tf.random_normal([D, K]),
                                          name="scale")))
            with tf.name_scope("qb_0"):
                qb_0 = Normal(loc=tf.Variable(tf.random_normal([K]), name="loc"),
                              scale=tf.nn.softplus(
                    tf.Variable(tf.random_normal([K]), name="scale")))

        with tf.name_scope("inference"):
            y_ph = tf.placeholder(tf.int32, [batch_size], name='out')
            inference = ABDivergence({W_0: qW_0, b_0: qb_0},
                                     data={y: y_ph})

        with tf.name_scope("optimizer"):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                       global_step,
                                                       decay_step_size,
                                                       decay_ratio,
                                                       staircase=True)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                               beta1=0.9,
                                               beta2=0.999,
                                               epsilon=10e-8)

    inference.initialize(optimizer=optimizer,
                         global_step=global_step,
                         n_samples=n_samples,
                         n_iter=n_iter,
                         n_print=10000,
                         alpha=alpha,
                         beta=beta)
    tf.global_variables_initializer().run()
    # create log writer object
    writer = tf.summary.FileWriter(logs_path,
                                   graph=tf.get_default_graph())

    # Let the training begin. We load the data in minibatches and update the VI infernce using each new batch.
    print(inference.n_iter)
    for _ in range(inference.n_iter):
        X_batch, Y_batch = next_batch_outliers(mnist.train, batch_size,
                                               p_out=p_outliers)
        info_dict = inference.update(feed_dict={X: X_batch, y_ph: Y_batch})
        inference.print_progress(info_dict)

    # TEST
    # Load the test images.
    X_test = mnist.test.images
    Y_test = mnist.test.labels

    n_samples = 100
    prob_lst = []

    for _ in range(n_samples):
        w0_samp = qW_0.sample()
        b0_samp = qb_0.sample()

        prob = neural_network(X_test, w0_samp, b0_samp)
        prob_lst.append(prob.eval())

    Y_pred = np.argmax(np.mean(prob_lst, axis=0), axis=1)
    acc = (Y_pred == Y_test).mean() * 100
    print("accuracy in predicting the test data = {}".format(acc))

    pd_info = [n_iter, batch_size, n_samples, p_outliers,
               alpha, beta, acc]

    out = out.append(pd.DataFrame([pd_info], columns=col))
    i += 1
    out.to_hdf(os.path.join(logs_path, output_name), 'table')

print(pd_info)
out.to_hdf(os.path.join(logs_path, output_name), 'table')
