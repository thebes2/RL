import tensorflow as tf
import sys
import os
import random
from datetime import datetime
from tqdm.auto import tqdm
import numpy as np

class DQN_agent:

    def __init__(self, 
                 model,
                 buffer,
                 target=None,
                 env=None,
                 mode='DQN',
                 learning_rate=0.001,
                 batch_size=100,
                 update_steps=5,
                 beta=0.05,
                 epsilon=0.1,
                 gamma=0.99,
                 env_name='',
                 algo_name='',
                 run_name=None,
                 ckpt_folder=None):
        self.model = model
        self.buffer = buffer
        self.env = env
        self.target = target if target is not None else model
        self.mode = mode

        self._stdout = sys.stdout

        self.lr = learning_rate
        self.batch_size = batch_size
        self.update_steps = update_steps
        self.beta = beta
        self.epsilon = epsilon
        self.gamma = gamma
    
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.env_name = env_name
        self.algo_name = algo_name
        self.run_name = run_name
        if self.run_name is None:
            now = datetime.now()
            self.run_name = "{}-{}-{}".format(env_name, algo_name, now.strftime('%H-%M-%S'))

        if ckpt_folder is None:
            self.ckpt_dir = os.path.join('checkpoints', self.run_name)
        else:
            self.ckpt_dir = os.path.join(ckpt_folder, self.run_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.ckpt = tf.train.Checkpoint(
            model=self.model,
            target=self.target,
            optimizer=self.optimizer
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, directory=self.ckpt_dir, max_to_keep=3
        )

        self.update_counter = 0

    def get_model(self, obs, batch=False):
        # in Q-learning, the model predicts Q-values rather than probabilities
        obs = obs if batch else np.array([obs])
        probs = self.model(obs)
        return probs

    def get_target(self, obs, batch=False):
        # in Q-learning, the model predicts Q-values rather than probabilities
        obs = obs if batch else np.array([obs])
        probs = self.target(obs)
        return probs
    
    def get_action(self, obs, mode='eps-greedy', **kwargs):
        # in DQN, default behaviour is epsilon greedy for training
        if mode not in ('greedy', 'eps-greedy', 'categorical'):
            raise Exception('get_action: Invalid mode.')
        q = self.get_model(obs, **kwargs)
        n, a = q.shape
        if mode == 'greedy':
            return np.expand_dims(np.argmax(q, axis=1), 1)
        elif mode == 'eps-greedy':
            if random.uniform(0, 1) < self.epsilon:
                return np.expand_dims(np.random.choice(a, size=n), 1)
            else:
                return np.expand_dims(np.argmax(q, axis=1), 1)
        raise NotImplementedException("Cannot sample directly from Q-values, maybe treat them as logits later?")
        return tf.random.categorical(tf.math.log(q), 1).numpy()

    def preprocess(self, obs):
        if self.env_name == 'snake':
            return np.array(obs.astype(np.float32)[::10,::10]/255.0)
        elif self.env_name == 'tetris':
            return np.array(obs.astype(np.float32)/255.0)
        elif self.env_name == 'taxi':
            return np.array([obs])
        return obs

    def action_wrapper(self, action):
        """Some environments take actions in weird formats"""
        if self.env_name == 'snake':
            return [action]
        return action

    def detach_stdout(self):
        f = open(os.devnull, 'w')
        sys.stdout = f
        
    def attach_stdout(self):
        sys.stdout = self._stdout

    def collect_rollout(self, t_max=10000, policy=None, silenced=True, train=False):
        if silenced: self.detach_stdout()
        obs = self.preprocess(self.env.reset())
        dn = False
        i = 0
        reward = 0
        while i != t_max:
            act = self.get_action(obs)[0][0] if policy is None else policy(obs)
            oo, rr, dn, info = self.env.step(self.action_wrapper(act))
            oo = self.preprocess(oo)
            self.buffer.add((obs, act, rr, 0.0 if dn else self.gamma, oo))
            reward += rr
            obs = oo

            if train:
                for _ in range(self.update_steps):
                    self.update_model_step()

            if dn:
                break
            i += 1
        if silenced: self.attach_stdout()
        return reward

    def discount_rewards(self, r):
        for i in range(len(r)-1, -1, -1):
            r[i] += 0 if i==len(r)-1 else self.gamma * r[i+1]
        return r

    def unpack(self, samples):
        s, a, r, p, ss = [], [], [], [], []
        for sample in samples:
            s.append(sample[0])
            a.append(sample[1])
            r.append(sample[2])
            p.append(sample[3])
            ss.append(sample[4])
        return np.array(s), np.array(a), np.array(r), np.array(p), np.array(ss)

    def compute_model_loss(self, s, a, r, p, ss):
        if 'DDQN' in self.mode:
            idxs = np.argmax(self.get_model(ss, batch=True), axis=1)
            q = self.get_target(ss, batch=True)
            idx = tf.one_hot(idxs, tf.shape(q)[1])
            y = tf.reduce_sum(tf.math.multiply(q, idx), axis=1)
        else:
            q = self.get_target(ss, batch=True)
            y = np.max(q, axis=1)
        y = r + tf.multiply(p, y)
        preds = self.get_model(s, batch=True)
        a = tf.one_hot(a, tf.shape(preds)[1])
        q_pred = tf.reduce_sum(tf.multiply(preds, a), axis=1)
        loss = tf.reduce_mean(tf.square(q_pred - y))
        return loss

    def update_model_step(self):
        s, a, r, p, ss = self.unpack(self.buffer.sample(self.batch_size))
        with tf.GradientTape() as tape:
            model_loss = self.compute_model_loss(
                s, a, tf.constant(r, dtype=tf.float32),
                tf.constant(p, dtype=tf.float32), ss
            )
        model_grads = tape.gradient(model_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(model_grads, self.model.trainable_variables))
        self.update_counter += 1
        if self.update_counter == 100:
            self.update_counter = 0
            self.update_target_step()

    def update_target_step(self):
        model_weights = self.model.get_weights()
        target_weights = self.target.get_weights()
        self.target.set_weights(list(map(
            lambda x: self.beta * x[0] + (1.0 - self.beta) * x[1],
            zip(model_weights, target_weights)
        )))

    # @tf.function(experimental_relax_shapes=True)
    def update_network(self, steps):
        for _ in range(steps):
            self.update_model_step()
        self.update_target_step()

    def train(self, epochs=1000, t_max=10000, logging=True):
        avg_reward = 0.
        epochs_per_log = 5
        for t in tqdm(range(epochs), desc='Training epochs'):
            avg_reward += self.collect_rollout(t_max=t_max, silenced=True, train=True)
            # self.update_network(self.update_steps)
            # self.update_target_step()

            if logging and t % epochs_per_log == epochs_per_log-1:
                avg_reward /= epochs_per_log
                print("[{}] Average reward: {}".format(t+1, avg_reward))
                print("Predicted reward: {}".format(self.get_model(self.preprocess(self.env.reset()))))
                avg_reward = 0
                self.save_to_checkpoint()

    def load(self, path):
        model_path = os.path.join(path, 'model.h5')
        self.model = tf.keras.models.load_model(model_path)

    def load_from_checkpoint(self): 
        self.ckpt_manager.restore_or_initialize()

    def save(self, path=''):
        if len(path) == 0:
            now = datetime.now()
            path = "{}{}-{}".format(
                self.algo_name, '-{}'.format(self.env_name) if len(self.env_name) > 0 else '',
                now.strftime("%H-%M-%S")
            )
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, 'model.h5')
        self.model.save(model_path)

    def save_to_checkpoint(self):
        print('Saving to checkpoint...')
        self.ckpt_manager.save()
