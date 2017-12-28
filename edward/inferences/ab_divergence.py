from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import numpy as np
import tensorflow as tf

from edward.inferences.variational_inference import VariationalInference
from edward.models import RandomVariable
from edward.util import copy

try:
  from edward.models import Normal
  from tensorflow.contrib.distributions import kl_divergence
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


class ABDivergence(VariationalInference):
  """Variational inference with the AB divergence
  """

  def __init__(self, *args, **kwargs):
    super(ABDivergence, self).__init__(*args, **kwargs)

    self.is_reparameterizable = all([
        rv.reparameterization_type ==
        tf.contrib.distributions.FULLY_REPARAMETERIZED
        for rv in six.itervalues(self.latent_vars)])

  def initialize(self, n_samples=32, batch_size=32,
                 alpha=.2, beta=0.5, *args, **kwargs):
    """
    """
    self.n_samples = n_samples
    self.alpha = alpha
    self.beta = beta

    return super(ABDivergence, self).initialize(*args, **kwargs)

  def build_loss_and_gradients(self, var_list):

    if self.is_reparameterizable:
      p_log_prob = [0.0] * self.n_samples
      q_log_prob = [0.0] * self.n_samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(self.n_samples):
        # Form dictionary in order to replace conditioning on prior or
        # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
          if isinstance(x, RandomVariable):
            if isinstance(qx, RandomVariable):
              qx_copy = copy(qx, scope=scope)
              dict_swap[x] = qx_copy.value()
            else:
              dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))

        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          p_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))

        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            p_log_prob[s] += tf.reduce_sum(
                self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))

      # Reduces to a Renyi divergence:
      if np.abs(self.alpha + self.beta - 1.0) < 10e-3:
        log_ratios = [p - q for p, q in zip(p_log_prob, q_log_prob)]

        if np.abs(self.alpha - 1.0) < 10e-3:
          loss = tf.reduce_mean(log_ratios)
        else:
          log_ratios = tf.stack(log_ratios)
          log_ratios = log_ratios * (1 - self.alpha)
          log_ratios_max = tf.reduce_max(log_ratios, 0)
          log_ratios = tf.log(
              tf.maximum(1e-9,
                         tf.reduce_mean(tf.exp(log_ratios - log_ratios_max),
                                        0)))
          log_ratios = (log_ratios + log_ratios_max) / (1 - self.alpha)
          loss = tf.reduce_mean(log_ratios)
        loss = -loss

      # AB-objective:
      else:
        # Case 1: alpha + beta = 0
        if np.abs(self.alpha - self.beta) < 10e-3:
          print("Case 1: alpha + beta = 0")
          log_ratios1 = tf.stack([self.alpha * (p - q)
                                  for p, q in zip(p_log_prob, q_log_prob)])
          log_ratios2 = tf.stack([(p - q)
                                  for p, q in zip(p_log_prob, q_log_prob)])

          log_ratios1_max = tf.reduce_max(log_ratios1, 0)
          log_ratios2_max = tf.reduce_max(log_ratios2, 0)

          log_ratios1 = tf.log(
              tf.maximum(1e-9,
                         tf.reduce_mean(tf.exp(log_ratios1 - log_ratios1_max), 0))) \
              + log_ratios1_max
          log_ratios2 = tf.maximum(1e-9,
                                   tf.reduce_mean(tf.exp(log_ratios2 - log_ratios2_max), 0)) \
              + log_ratios2_max

          log_ratios = \
              log_ratios1 / (self.beta * (self.alpha + self.beta)) \
              + log_ratios2 / (self.alpha * (self.alpha + self.beta)) \
              - log_ratios3 / (self.alpha * self.beta)

          log_ratios = tf.maximum(1.e-9, log_ratios)
          loss = tf.reduce_mean(log_ratios)

        # Case 2: alpha = 0, beta != 0
        elif np.abs(self.alpha) < 10e-3 and np.abs(self.beta) > 10e-3:
          print("Case 2: alpha = 0, beta != 0")
          log_ratios = tf.stack([tf.exp(self.beta * q) * (q - p)
                                 for p, q in zip(p_log_prob, q_log_prob)])

          log_ratios = log_ratios / self.beta
          log_ratios = tf.maximum(1.e-9, log_ratios)
          loss = tf.reduce_mean(log_ratios)

        # Case 3: alpha != 0, beta = 0
        elif np.abs(self.beta) < 10e-3 and np.abs(self.alpha) > 10e-3:
          print("Case 2: alpha != 0, beta = 0")
          log_ratios = tf.stack([tf.exp(self.alpha * p) * (p - q)
                                 for p, q in zip(p_log_prob, q_log_prob)])

          log_ratios = log_ratios / self.alpha
          log_ratios = tf.maximum(1.e-9, log_ratios)
          loss = tf.reduce_mean(log_ratios)

        # Case 4: alpha = 0, beta = 0
        elif np.abs(self.beta) < 10e-3 and np.abs(self.alpha) < 10e-3:
          print("Case 4: alpha = 0, beta = 0")
          log_ratios = tf.stack([(p - q)**2
                                 for p, q in zip(p_log_prob, q_log_prob)])

          log_ratios = log_ratios / 2.
          log_ratios = tf.maximum(1.e-9, log_ratios)
          loss = tf.reduce_mean(log_ratios)

        # Case 5: Normal case:
        else:
          print("Case 5: Normal")
          log_ratios1 = tf.stack([(self.alpha + self.beta - 1) * q
                                  for q in q_log_prob])
          log_ratios2 = tf.stack([(self.alpha + self.beta) * p - q
                                  for p, q in zip(p_log_prob, q_log_prob)])
          log_ratios3 = tf.stack([self.beta * p - (1 - self.alpha) * q
                                  for p, q in zip(p_log_prob, q_log_prob)])

          log_ratios1_max = tf.reduce_max(log_ratios1, 0)
          log_ratios2_max = tf.reduce_max(log_ratios2, 0)
          log_ratios3_max = tf.reduce_max(log_ratios3, 0)

          log_ratios1 = tf.log(
              tf.maximum(1e-9,
                tf.reduce_mean(tf.exp(log_ratios1 - log_ratios1_max), 0))) \
                + log_ratios1_max

          log_ratios2 = tf.log(
            tf.maximum(1e-9,
              tf.reduce_mean(tf.exp(log_ratios2 - log_ratios2_max), 0))) \
              + log_ratios2_max

          log_ratios3 = tf.log(
            tf.maximum(1e-9,
              tf.reduce_mean(tf.exp(log_ratios3 - log_ratios3_max), 0))) \
              + log_ratios3_max

          log_ratios = \
            log_ratios1 / (self.beta * (self.alpha + self.beta)) \
            + log_ratios2 / (self.alpha * (self.alpha + self.beta)) \
            - log_ratios3 / (self.alpha * self.beta)

          log_ratios = tf.maximum(1.e-9, log_ratios)
          loss = tf.reduce_mean(log_ratios)

      if self.logging:
        p_log_prob = tf.reduce_mean(p_log_prob)
        q_log_prob = tf.reduce_mean(q_log_prob)
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])

      grads = tf.gradients(loss, var_list)
      grads_and_vars = list(zip(grads, var_list))
      return loss, grads_and_vars
    else:
      raise NotImplementedError(
          "Variational AB inference only works with reparameterizable models")
