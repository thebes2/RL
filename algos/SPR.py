import tensorflow as tf
import numpy as np
import random
import math

from algos.DQN import DQN_agent
from rl.models import (
    get_vision_architecture, 
    get_qlearning_architecture, 
    get_transition_architecture, 
    get_projection_architecture, 
    get_prediction_architecture
)
from utils.Buffer import ReplayBuffer



class SPR_agent(DQN_agent):

    """
    Temporary class for implementing DQN augmented with self-predicted representations
    TODO: Merge functionality back into DQN through trainers/losses
    """

    def __init__(self, config):
        super(SPR_agent, self).__init__(config)

        self.lam = config['lambda']

        self.vision = get_vision_architecture(config['env'], algo=config['algo'])
        self.vision_target = get_vision_architecture(config['env'], algo=config['algo'])
        self.qlearning = get_qlearning_architecture(config['env'], algo=config['algo'])
        self.transition = get_transition_architecture(config['env'])
        self.projection = get_projection_architecture(config['env'], cfg=config)
        self.projection_target = get_projection_architecture(config['env'], cfg=config)
        self.prediction = get_prediction_architecture(config['env'], cfg=config)

        # TODO: change the default 'multistep' to always be 1
        self.buffer = ReplayBuffer(num_samples=config['batch_size'], mode='spr', steps=config['_multistep'])

        self.trainable_variables = sum(
            (model.trainable_variables for model in (
                self.vision,
                self.qlearning,
                self.transition,
                self.projection,
                self.prediction
            )), []
        )

        for callback in self.callbacks: # hack
            callback.on_init(self)

    def get_model(self, obs, batch=False):
        obs = obs if batch else np.array([obs])
        preds = self.qlearning(self.vision(obs))
        return preds


    def get_action(self, obs, mode='eps-greedy', **kwargs):
        q = self.get_model(obs, **kwargs)
        n, a = q.shape
        if mode == 'greedy':
            return np.expand_dims(np.argmax(q, axis=1), 1)
        elif mode == 'eps-greedy':
            if random.uniform(0, 1) < self.epsilon:
                return np.expand_dims(np.random.choice(a, size=n), 1)
            else:
                return np.expand_dims(np.argmax(q, axis=1), 1)
    
    
    def extract_from_trajectory(self, samples):
        s, a, r, p, ss = [], [], [], [], []
        for sample in samples:
            s.append(sample[0][0])
            a.append(sample[0][1])
            r.append(sum(step[2] for step in sample))
            p.append(math.prod(step[3] for step in sample))
            ss.append(sample[-1][4])
        return np.array(s), np.array(a), np.array(r), np.array(p), np.array(ss)


    def compute_rl_loss(self, s, a, r, p, ss, w=None):
        return super().compute_model_loss(s, a, r, p, ss, w)


    def compute_single_loss(self, trajectory):
        z_0 = self.vision(np.array([trajectory[0][0]]))
        cond = lambda i, _, __: i < len(trajectory)
        def step(i, z, l):
            s = trajectory[i]
            oh = np.eye(self.config['n_actions'])[trajectory[i-1][1]]
            oh = np.array([oh])
            inp = np.concatenate((z, oh), axis=1)
            z_pred = self.transition(inp)
            z_target = self.vision_target(np.array([s[0]]))
            y_pred = self.prediction(self.projection(z_pred))
            y_target = self.projection_target(z_target)
            y_pred = y_pred / tf.sqrt(tf.reduce_sum(tf.square(y_pred), 1))
            y_target = y_target / tf.sqrt(tf.reduce_sum(tf.square(y_target), 1))
            return i+1, z_pred, l-tf.linalg.matmul(y_pred, y_target, transpose_b=True)
        _, _, loss = tf.while_loop(cond, step, (1, z_0, 0.))
        return loss / len(trajectory) / self.config['latent_dim']


    def compute_model_loss(self, samples):
        loss = 0.
        for sample in samples:
            loss += self.compute_single_loss(sample)
        return loss


    def update_model_step(self):
        samples = self.buffer.sample(self.batch_size)
        s, a, r, p, ss = self.extract_from_trajectory(samples)
        with tf.GradientTape() as tape:
            rl_loss = self.compute_rl_loss(s, a, r, p, ss)
            spr_loss = self.compute_model_loss(samples)[0][0]
            print(rl_loss, spr_loss)
            loss = rl_loss + self.lam * spr_loss

        grads = tape.gradient(loss, self.trainable_variables)
        clipped_grads = [tf.clip_by_value(grad, -1., 1.) for grad in grads]
        self.optimizer.apply_gradients(zip(clipped_grads, self.trainable_variables))
        self.update_target_step()

    
    def average_weights(self, model, target, delta):
        model_weights = model.get_weights()
        target_weights = target.get_weights()
        target.set_weights(list(map(
            lambda x: delta * x[0] + (1. - delta) * x[1],
            zip(model_weights, target_weights)
        )))

    def update_target_step(self):
        self.average_weights(self.vision, self.vision_target, self.delta)
        self.average_weights(self.projection, self.projection_target, self.delta)