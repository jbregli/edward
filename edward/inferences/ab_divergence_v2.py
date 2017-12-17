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
    raise ImportError(
        "{0}. Your TensorFlow version is not supported.".format(e))


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
            alpha = [self.alpha] * self.n_samples
            beta = [self.beta] * self.n_samples
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

            # KL:
            kl_penalty = tf.reduce_sum([
                tf.reduce_sum(kl_divergence(qz, z))
                for z, qz in six.iteritems(self.latent_vars)])

            # Regularization:
            loss_ab = -1 / (self.alpha * self.beta) * tf.reduce_sum(
                tf.exp([b * p for b, q in zip(beta, q_log_prob)])) + \
                + 1 / ((self.alpha + self.beta) * self.beta) * tf.reduce_sum(
                    tf.exp([(a + b) * q for a, b, p
                            in zip(alpha, beta, p_log_prob)]))

            loss = -kl_penalty + loss_ab

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
