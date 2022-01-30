import numpy as np
import tensorflow as tf

from .Agent import RL_agent


class PPO_agent(RL_agent):
    def __init__(self, policy_net, value_net, epsilon=0.2, **kwargs):

        super(PPO_agent, self).__init__(
            policy_net,
            value_net,
            algo_name="PPO",
            **kwargs,
        )

        self.epsilon = epsilon

    def compute_policy_loss(self, obs, act, val, **kwargs):
        probs = kwargs["probs"]
        adv = kwargs["adv"]
        policy = self.get_policy(obs, batch=True)
        ind = tf.one_hot(tf.squeeze(act), policy.shape[1])
        new_probs = tf.reduce_sum(tf.multiply(policy, ind), axis=1)
        improve = tf.math.multiply(tf.math.divide(new_probs, probs), adv)
        clip = tf.where(
            adv >= 0,
            tf.multiply(adv, 1 + self.epsilon),
            tf.multiply(adv, 1 - self.epsilon),
        )
        stk = tf.stack([improve, clip])
        res = tf.math.reduce_min(stk, axis=0)
        return -tf.math.reduce_mean(res)

    @tf.function(experimental_relax_shapes=True)
    def update_network(self, obs, act, val, **kwargs):
        policy = self.get_policy(obs, batch=True)
        ind = tf.one_hot(tf.squeeze(act), policy.shape[1])
        probs = tf.reduce_sum(tf.multiply(policy, ind), axis=1)
        adv = val - tf.squeeze(self.get_value(obs, batch=True))
        for _ in range(self.gradient_steps):
            self.update_policy_step(obs, act, val, probs=probs, adv=adv, **kwargs)
        for _ in range(self.gradient_steps):
            self.update_value_step(obs, act, val, **kwargs)

    # @tf.function(experimental_relax_shapes=True)
    def mpi_update_network(self, obs, act, val, rnk, **kwargs):
        policy = self.get_policy(obs, batch=True)
        ind = tf.one_hot(tf.squeeze(act), policy.shape[1])
        probs = tf.reduce_sum(tf.multiply(policy, ind), axis=1)
        adv = val - tf.squeeze(self.get_value(obs, batch=True))
        for _ in range(self.gradient_steps):
            self.mpi_update_policy_step(
                obs, act, val, rnk, probs=probs, adv=adv, **kwargs
            )
        for _ in range(self.gradient_steps):
            self.mpi_update_value_step(obs, act, val, rnk, **kwargs)
