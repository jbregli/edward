"""Convolutional variational auto-encoder using AB-divergence

The neural networks are written with TensorFlow Slim.

References
----------
http://edwardlib.org/tutorials/decoder
http://edwardlib.org/tutorials/inference-networks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import edward as ed
import numpy as np
import os
ROOT_DIR = os.environ['ROOT_DIR']
import sys
sys.path.append(ROOT_DIR)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_curve, auc

from edward.models import Bernoulli, Normal
from edward.util import Progbar
from observations import mnist
from scipy.misc import imsave
from tensorflow.contrib import slim

from edward.inferences.ab_divergence import ABDivergence

from tensorflow.examples.tutorials.mnist import input_data

from os.path import join as ojoin

#############
### UTILS ###
#############


def generative_network(z):
    """Generative network to parameterize generative model. It takes
    latent variables as input and outputs the likelihood parameters.

    logits = neural_network(z)
    """
    with slim.arg_scope([slim.conv2d_transpose],
                        activation_fn=tf.nn.elu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'scale': True}):
        net = tf.reshape(z, [M, 1, 1, d])
        net = slim.conv2d_transpose(net, 128, 3, padding='VALID')
        net = slim.conv2d_transpose(net, 64, 5, padding='VALID')
        net = slim.conv2d_transpose(net, 32, 5, stride=2)
        net = slim.conv2d_transpose(net, 1, 5, stride=2, activation_fn=None)
        net = slim.flatten(net)
    return net


def inference_network(x):
    """Inference network to parameterize variational model. It takes
    data as input and outputs the variational parameters.

    loc, scale = neural_network(x)
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.elu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'scale': True}):
        net = tf.reshape(x, [M, 28, 28, 1])
        net = slim.conv2d(net, 32, 5, stride=2)
        net = slim.conv2d(net, 64, 5, stride=2)
        net = slim.conv2d(net, 128, 5, padding='VALID')
        net = slim.dropout(net, 0.9)
        net = slim.flatten(net)
        params = slim.fully_connected(net, d * 2, activation_fn=None)

    loc = params[:, :d]
    scale = tf.nn.softplus(params[:, d:])
    return loc, scale


def reconstruction_error(x_recon, x, avg=False):
    """
    Compute reconstruction error
    """
    error = np.mean(np.abs(x_recon - x), axis=1)
    if avg:
        error = np.mean(error, axis=0)
    return error


def reconstruction_score(generator, p, num_examples, sess, batch_size=128, avg=True):
    n_iter = num_examples // batch_size
    score = []
    for t in range(1, n_iter + 1):
        test_data = generator.next(batch_size=batch_size, p=p)

        test_logits = sess.run(logits, {x_ph: test_data})
        test_hidden_rep = tf.sigmoid(test_logits)
        # Visualize hidden representations.
        test_reconstruction = test_hidden_rep.eval()

        score.append(reconstruction_error(
            test_data, test_reconstruction, avg=False))

    score = np.concatenate(score)

    if avg:
        score = score.mean()

    return score


class OutlierGenerator(object):

    def __init__(self, inlier, outlier):
        """

        """
        self.inlier = inlier
        self.outlier = outlier

    def next(self, batch_size=32, p=0.0):

        # Idx to be swapped
        idx = np.random.binomial(1, p=p, size=batch_size)

        outliers, _ = self.outlier.next_batch(np.sum(idx))
        inliers, _ = self.inlier.next_batch(batch_size)

        batch = copy.deepcopy(inliers)
        batch[np.where(idx), :] = copy.deepcopy(outliers)

        return batch


##############
### PARAMS ###
##############
# Data
inliers_dir = ojoin(ROOT_DIR, "data", "raw", "MNIST")
outliers_dir = ojoin(ROOT_DIR, "data", "interim", "notMNIST_data")
out_dir = ojoin(ROOT_DIR, "data", "processed", "anomaly_detection")

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

inliers_data = input_data.read_data_sets(inliers_dir, one_hot=True)
outliers_data = input_data.read_data_sets(outliers_dir, one_hot=True)

outlier_gen = OutlierGenerator(inliers_data.train, outliers_data.train)

# Meta-parameters
M = 128         # batch size during training
d = 10          # latent dimension
p = 0.0         # probability of outliers
n_epoch = 100   # Number of training epochs
alpha = 1.0     # AB-divergence parameter 1
beta = 0.5      # AB-divergence parameter 2
n_samples = 50  # AB-divergence number of n_samples

starter_learning_rate = 0.01
decay_ratio = 0.9
decay_step_size = 25


out_dir_training = os.path.join(out_dir, 'training',
                                'a{}_b{}_p{}_K{}_n{}'.format(alpha, beta,
                                                             p, n_samples,
                                                             n_epoch))
out_dir_inliers = os.path.join(out_dir, 'inliers',
                               'a{}_b{}_p{}_K{}_n{}'.format(alpha, beta,
                                                            p, n_samples,
                                                            n_epoch))
out_dir_outliers = os.path.join(out_dir, 'outliers',
                                'a{}_b{}_p{}_K{}_n{}'.format(alpha, beta,
                                                             p, n_samples,
                                                             n_epoch))
if not os.path.exists(out_dir_training):
    os.makedirs(out_dir_training)
if not os.path.exists(out_dir_inliers):
    os.makedirs(out_dir_inliers)
if not os.path.exists(out_dir_outliers):
    os.makedirs(out_dir_outliers)


#############
### MODEL ###
#############
z = Normal(loc=tf.zeros([M, d]), scale=tf.ones([M, d]))
logits = generative_network(z)
x = Bernoulli(logits=logits)

#################
### INFERENCE ###
#################
# INFERENCE
x_ph = tf.placeholder(tf.int32, [M, 28 * 28])
loc, scale = inference_network(tf.cast(x_ph, tf.float32))
qz = Normal(loc=loc, scale=scale)

# Bind p(x, z) and q(z | x) to the same placeholder for x.
data = {x: x_ph}
# inference = ed.KLqp({z: qz}, data)
inference = ABDivergence({z: qz}, data)

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

optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
inference.initialize(optimizer=optimizer, global_step=global_step,
                     n_samples=n_samples, alpha=alpha, beta=beta)

hidden_rep = tf.sigmoid(logits)

tf.global_variables_initializer().run()

n_iter_per_epoch = inliers_data.train.num_examples // M
for epoch in range(1, n_epoch + 1):
    print("Epoch: {0}".format(epoch))
    avg_loss = 0.0

    pbar = Progbar(n_iter_per_epoch)
    for t in range(1, n_iter_per_epoch + 1):
        pbar.update(t)
        x_batch = outlier_gen.next(batch_size=M, p=p)
        info_dict = inference.update(feed_dict={x_ph: x_batch})
        avg_loss += info_dict['loss']

    # Print a lower bound to the average marginal likelihood for an
    # image.
    avg_loss = avg_loss / n_iter_per_epoch
    avg_loss = avg_loss / M
    print("D(p(x)||q(x) ~ {:0.5f}".format(avg_loss))

    # Visualize hidden representations.
    images = hidden_rep.eval()
    for m in range(M):
        imsave(os.path.join(out_dir_training, '%d.png') %
               m, images[m].reshape(28, 28))

##################
### PREDICTION ###
##################
sess = ed.get_session()
outlier_gen_test = OutlierGenerator(inliers_data.test, outliers_data.test)
inliers_batch_test = outlier_gen_test.next(batch_size=M, p=0)
outliers_batch_test = outlier_gen_test.next(batch_size=M, p=1)

# Inliers reconstruction:
inliers_test_logits = sess.run(logits, {x_ph: inliers_batch_test})
inliers_test_hidden_rep = tf.sigmoid(inliers_test_logits)
inliers_test_images = inliers_test_hidden_rep.eval()
for m in range(M):
    imsave(os.path.join(out_dir_inliers, '%d.png') % m,
           inliers_test_images[m].reshape(28, 28))

# Outlier reconstruction
outliers_test_logits = sess.run(logits, {x_ph: outliers_batch_test})
outliers_test_hidden_rep = tf.sigmoid(outliers_test_logits)
outliers_test_images = outliers_test_hidden_rep.eval()
for m in range(M):
    imsave(os.path.join(out_dir_outliers, '%d.png') % m,
           outliers_test_images[m].reshape(28, 28))

# Reconstruction scores:
inliers_score = reconstruction_score(outlier_gen_test, p=0,
                                     num_examples=outlier_gen_test.inlier.num_examples,
                                     sess=sess, batch_size=128, avg=False)
print("Average inliers reconstruction test score: {}".format(inliers_score.mean()))
outliers_score = reconstruction_score(outlier_gen_test, p=1,
                                      num_examples=outlier_gen_test.outlier.num_examples,
                                      sess=sess, batch_size=128, avg=False)
print("Average outliers reconstruction test score: {}".format(outliers_score.mean()))

# ROC curve:
gt = np.concatenate([np.zeros(inliers_score.shape),
                     np.ones(outliers_score.shape)])
fpr, tpr, threshold = roc_curve(gt,
                                np.concatenate([inliers_score, outliers_score]))
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig(ojoin(out_dir,
                  'roc_curve_a{}_b{}_p{}_K{}_n{}.pdf'.format(alpha, beta,
                                                             p, n_samples,
                                                             n_epoch)),
            format='pdf')
