import os
import sys
from datetime import datetime
from multiprocessing.pool import ThreadPool

import numpy as np
import tensorflow as tf
from mpi4py import MPI
from tqdm.auto import tqdm

from utils.mpi import average_gradients, broadcast_model, get_size, mpi_print

sys.path.insert(0, "..")


class RL_agent:
    def __init__(
        self,
        policy_net,
        value_net,
        env=None,
        make_env=None,
        learning_rate=1e-4,
        minibatch_size=100,
        gradient_steps=5,
        gamma=0.99,
        threads=1,
        env_name="",
        algo_name="",
        run_name=None,
        ckpt_folder=None,
    ):

        self.policy = policy_net
        self.value = value_net

        self._stdout = sys.stdout

        if env is None and make_env is None:
            print(
                "No env or make_env provided. Can only run inference.", file=sys.stderr
            )

        self.threads = threads
        if threads == 1:
            if env is None:
                self.env = make_env() if make_env is not None else None
            else:
                self.env = env
        elif threads > 1:
            if make_env is None:
                raise Exception("Must supply make_env if multithreading.")
            self.env = [make_env() for i in range(threads)]

        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.gradient_steps = gradient_steps
        self.gamma = gamma

        self.policy_optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        self.value_optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

        self.env_name = env_name
        self.algo_name = algo_name
        self.run_name = run_name
        if self.run_name is None:
            now = datetime.now()
            self.run_name = "{}-{}-{}".format(
                env_name, algo_name, now.strftime("%H-%M-%S")
            )

        if ckpt_folder is None:
            self.ckpt_dir = os.path.join("checkpoints", self.run_name)
        else:
            self.ckpt_dir = os.path.join(ckpt_folder, self.run_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.ckpt = tf.train.Checkpoint(
            policy=self.policy,
            value=self.value,
            policy_optimizer=self.policy_optimizer,
            value_optimizer=self.value_optimizer,
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, directory=self.ckpt_dir, max_to_keep=3
        )

    def get_policy(self, obs, batch=False):
        obs = obs if batch else np.array([obs])
        probs = self.policy(obs)
        return probs

    def get_action(self, obs, greedy=False, **kwargs):
        probs = self.get_policy(obs, **kwargs)
        n, a = probs.shape
        if greedy:
            return np.argmax(probs, axis=1)
        return tf.random.categorical(tf.math.log(probs), 1).numpy()

    def get_value(self, obs, batch=False):
        obs = obs if batch else np.array([obs])
        return self.value(obs)

    def compute_policy_loss(self, obs, act, val, **kwargs):
        policy = self.get_policy(obs, batch=True)
        ind = tf.one_hot(tf.squeeze(act), policy.shape[1])
        adv = val - tf.squeeze(self.get_value(obs, batch=True))
        logprobs = tf.math.log(tf.reduce_sum(tf.multiply(policy, ind), axis=1))
        return -tf.math.reduce_mean(logprobs * adv)

    def update_policy_step(self, obs, act, val, **kwargs):
        with tf.GradientTape() as policy_tape:
            policy_loss = self.compute_policy_loss(obs, act, val, **kwargs)
        policy_grads = policy_tape.gradient(
            policy_loss, self.policy.trainable_variables
        )
        self.policy_optimizer.apply_gradients(
            zip(policy_grads, self.policy.trainable_variables)
        )

    def mpi_update_policy_step(self, obs, act, val, rnk, **kwargs):
        with tf.GradientTape() as policy_tape:
            policy_loss = self.compute_policy_loss(obs, act, val, **kwargs)
        policy_grads = policy_tape.gradient(
            policy_loss, self.policy.trainable_variables
        )
        avg_policy_grads = []
        for grad in policy_grads:
            MPI.COMM_WORLD.Barrier()
            shape = grad.shape
            grad = average_gradients(grad.numpy().flatten())
            avg_policy_grads.append(grad.reshape(shape))
        self.policy_optimizer.apply_gradients(
            zip(policy_grads, self.policy.trainable_variables)
        )

    def compute_value_loss(self, obs, act, val, **kwargs):
        preds = tf.squeeze(self.get_value(obs, batch=True))
        return tf.reduce_mean(tf.square(preds - val))

    def update_value_step(self, obs, act, val, **kwargs):
        with tf.GradientTape() as value_tape:
            value_loss = self.compute_value_loss(obs, act, val, **kwargs)
        value_grads = value_tape.gradient(value_loss, self.value.trainable_variables)
        self.value_optimizer.apply_gradients(
            zip(value_grads, self.value.trainable_variables)
        )

    def mpi_update_value_step(self, obs, act, val, rnk, **kwargs):
        with tf.GradientTape() as value_tape:
            value_loss = self.compute_value_loss(obs, act, val, **kwargs)
        value_grads = value_tape.gradient(value_loss, self.value.trainable_variables)
        avg_value_grads = []
        for grad in value_grads:
            MPI.COMM_WORLD.Barrier()
            shape = grad.shape
            grad = average_gradients(grad.numpy().flatten())
            avg_value_grads.append(grad.reshape(shape))
        self.value_optimizer.apply_gradients(
            zip(value_grads, self.value.trainable_variables)
        )

    def preprocess(self, obs):
        if self.env_name == "snake":
            return np.array(obs.astype(np.float32)[::10, ::10] / 255.0)
        elif self.env_name == "tetris":
            return np.array(obs.astype(np.float32) / 255.0)
        elif self.env_name == "taxi":
            return np.array([obs])
        return obs

    def action_wrapper(self, action):
        """Some environments take actions in weird formats"""
        if self.env_name == "snake":
            return [action]
        return action

    def detach_stdout(self):
        f = open(os.devnull, "w")
        sys.stdout = f

    def attach_stdout(self):
        sys.stdout = self._stdout

    def collect_rollout(self, t_max=10000, policy=None, silenced=True, display=False):
        if silenced:
            self.detach_stdout()
        obs = self.preprocess(self.env.reset())
        if display:
            self.env.render()
        o, a, r = [], [], []
        dn = False
        i = 0
        while i != t_max:
            act = self.get_action(obs)[0][0] if policy is None else policy(obs)
            oo, rr, dn, info = self.env.step(self.action_wrapper(act))
            if display:
                self.env.render()
            o.append(obs)
            a.append(act)
            r.append(rr)
            obs = self.preprocess(oo)

            if dn:
                break
            i += 1
        if silenced:
            self.attach_stdout()
        return o, a, r

    def discount_rewards(self, r):
        for i in range(len(r) - 1, -1, -1):
            r[i] += 0 if i == len(r) - 1 else self.gamma * r[i + 1]
        return r

    @tf.function(experimental_relax_shapes=True)
    def update_network(self, obs, act, val, **kwargs):
        self.update_policy_step(obs, act, val, **kwargs)

        for _ in range(self.gradient_steps):
            self.update_value_step(obs, act, val, **kwargs)

    # @tf.function(experimental_relax_shapes=True)
    def mpi_update_network(self, obs, act, val, rnk, **kwargs):
        self.mpi_update_policy_step(obs, act, val, rnk, **kwargs)

        for _ in range(self.gradient_steps):
            self.mpi_update_value_step(obs, act, val, rnk, **kwargs)

    def warmup(self, n_roll=1000, t_steps=5, pre_epochs=10, examples=None, t_max=10000):
        """Although these algos are on-policy, we can train on good examples to initialize the policy network"""
        self.train(pre_epochs, t_max, logging=False)
        obs, act, val = [], [], []
        if examples:
            obs = examples.get("observations", [])
            act = examples.get("actions", [])
            val = examples.get("rewards", [])
        for i in tqdm(range(n_roll), desc="Collecting trajectories"):
            o, a, v = self.collect_rollout(t_max=t_max)
            v = self.discount_rewards(v)
            obs.extend(o)
            act.extend(a)
            val.extend(v)

        for i in range(t_steps):
            self.update_policy(obs, act, val)

    def train(
        self,
        epochs=1000,
        t_max=10000,
        logging=True,
        buf_size=10000,
        min_buf_size=1000,
        display=False,
    ):
        avg_reward = 0.0
        epochs_per_log = 5
        for t in tqdm(range(epochs), desc="Training epochs"):
            obs, act, val = [], [], []
            for _ in range(self.minibatch_size):
                o, a, v = self.collect_rollout(
                    t_max=t_max, silenced=True, display=display
                )
                avg_reward += sum(v)
                v = self.discount_rewards(v)
                obs.extend(o)
                act.extend(a)
                val.extend(v)

                if len(obs) > buf_size:
                    self.update_network(
                        tf.constant(obs),
                        tf.constant(act),
                        tf.constant(val, dtype=tf.float32),
                    )
                    obs, act, val = [], [], []

            if len(obs) > min_buf_size:
                self.update_network(
                    tf.constant(obs),
                    tf.constant(act),
                    tf.constant(val, dtype=tf.float32),
                )
            if logging and t % epochs_per_log == epochs_per_log - 1:
                avg_reward /= epochs_per_log * self.minibatch_size
                print("[{}] Average reward: {}".format(t + 1, avg_reward))
                avg_reward = 0
                self.save_to_checkpoint()

    def mpi_train(self, rnk, epochs=1000, t_max=10000, logging=True, buf_size=20000):
        broadcast_model(self.policy)
        broadcast_model(self.value)
        buf_size = buf_size // get_size() + (1 if buf_size % get_size() > 0 else 0)

        avg_reward = 0.0
        cnt = 0
        epochs_per_log = 5

        for t in tqdm(range(epochs), desc="Training epochs", disable=(rnk > 0)):
            obs, act, val = [], [], []
            while len(obs) < buf_size:
                o, a, v = self.collect_rollout(t_max=t_max, silenced=True)
                avg_reward += sum(v)
                v = self.discount_rewards(v)
                obs.extend(o)
                act.extend(a)
                val.extend(v)
                cnt += 1

            self.mpi_update_network(
                tf.constant(obs), tf.constant(act), tf.constant(val), rnk
            )
            if logging and t % epochs_per_log == epochs_per_log - 1 and rnk == 0:
                avg_reward /= cnt
                mpi_print("[{}] Average reward: {}".format(t + 1, avg_reward))
                avg_reward, cnt = 0, 0
                self.save_to_checkpoint()
            sys.stdout.flush()

    def load(self, path):
        policy_path = os.path.join(path, "policy.h5")
        value_path = os.path.join(path, "value.h5")

        self.policy = tf.keras.models.load_model(policy_path)
        self.value = tf.keras.models.load_model(value_path)

    def load_from_checkpoint(self):
        self.ckpt_manager.restore_or_initialize()

    def save(self, path=""):
        if len(path) == 0:
            now = datetime.now()
            path = "{}{}-{}".format(
                self.algo_name,
                "-{}".format(self.env_name) if len(self.env_name) > 0 else "",
                now.strftime("%H-%M-%S"),
            )
        os.makedirs(path, exist_ok=True)
        policy_path = os.path.join(path, "policy.h5")
        value_path = os.path.join(path, "value.h5")

        self.policy.save(policy_path)
        self.value.save(value_path)

    def save_to_checkpoint(self):
        print("Saving to checkpoint...")
        self.ckpt_manager.save()
