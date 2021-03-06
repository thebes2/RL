import os
import random
import sys
from datetime import datetime
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from rl.model_builder import build_policy
from rl.models import get_policy_architecture
from utils.Buffer import ReplayBuffer, Transition
from utils.Callbacks import get_callbacks
from utils.Env import get_env
from utils.logger import logger
from utils.Trainer import l2_loss


class DQN_agent:
    def __init__(self, config):
        # hacks for now
        if "n_frames" not in config:
            config["n_frames"] = 1
        if "lambda" not in config:
            config["lambda"] = 0
        self.config = config

        self.model: tf.keras.Model = build_policy(config)
        self.buffer = ReplayBuffer(
            env=config["env_name"],
            max_size=config["max_buf_size"],
            mode="proportional" if "PER" in config["algo"] else "uniform",
        )
        self.env = get_env(config["env"], config["use_raw_env"])
        self.target = tf.keras.models.clone_model(self.model)
        self.algo = config["algo"]

        self._stdout = sys.stdout

        self.learning_rate = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.update_steps = config["update_steps"]
        self.update_freq = config["update_freq"]
        self.target_delay = config["target_delay"]
        self.multistep = config["multistep"]
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.delta = config["delta"]
        self.epsilon = config["epsilon"]
        self.gamma = config["gamma"]
        self.n_actions = config["n_actions"]
        self.t_max = config["t_max"]

        self.callbacks = get_callbacks(config.get("callbacks", []))

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, clipnorm=1.0
        )

        self.env_name = config["env_name"]
        self.raw_env_name = config.get("env", self.env_name)
        self.algo_name = str(config["algo"])
        self.run_name = config["run_name"]
        self.training = True
        self.existing = False

        if "ckpt_folder" not in config:
            self.ckpt_dir = os.path.join("checkpoints", self.run_name)
        else:
            self.ckpt_dir = os.path.join(config["ckpt_folder"], self.run_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.ckpt = tf.train.Checkpoint(
            model=self.model, target=self.target, optimizer=self.optimizer
        )  # self.checkpoint()
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, directory=self.ckpt_dir, max_to_keep=5
        )
        self.load_from_checkpoint(manual=False)

        self.update_counter = 0

        self.double_DQN = "DDQN" in self.algo
        self.prioritized_sampling = "PER" in self.algo
        self.noisy_DQN = "noisy" in self.algo

        self.counter = 0

    # the old init method for backwards compatibility
    def init(
        self,
        model,
        buffer,
        target=None,
        env=None,
        mode=("DQN"),
        learning_rate=0.001,
        batch_size=128,
        update_steps=5,
        update_freq=4,
        target_delay=500,
        multistep=1,
        alpha=1.0,
        beta=1.0,
        delta=0.005,
        epsilon=0.1,
        gamma=0.99,
        env_name="",
        algo_name="",
        run_name=None,
        ckpt_folder=None,
        callbacks=[],
        **kwargs,
    ):
        self.model = model
        self.buffer = buffer
        self.env = env
        self.target = target if target is not None else model
        self.mode = mode

        self._stdout = sys.stdout

        self.lr = learning_rate
        self.batch_size = batch_size
        self.update_steps = update_steps
        self.update_freq = update_freq
        self.target_delay = target_delay
        self.multistep = multistep
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.epsilon = epsilon
        self.gamma = gamma

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, clipnorm=1.0
        )

        self.env_name = env_name
        self.algo_name = algo_name
        self.run_name = run_name
        self.callbacks = callbacks

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

        self.ckpt_manager = None  # delay creation to allow subclass checkpointing

        self.update_counter = 0

        self.double_DQN = "DDQN" in self.mode
        self.prioritized_sampling = "PER" in self.mode
        self.noisy_DQN = "noisy" in self.mode

    def checkpoint(self) -> tf.train.Checkpoint:
        """Defines the tf objects to be tracked and checkpointed"""
        return tf.train.Checkpoint(
            model=self.model,
            target=self.target,
            optimizer=self.optimizer,
        )

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def set_model(self, model):
        self.model = model

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

    def get_action(self, obs, mode="eps-greedy", **kwargs):
        # in DQN, default behaviour is epsilon greedy for training
        if mode not in ("greedy", "eps-greedy", "categorical"):
            raise Exception("get_action: Invalid mode.")
        q = self.get_model(obs, **kwargs)
        n, a = q.shape
        if mode == "greedy":
            return np.expand_dims(np.argmax(q, axis=1), 1)
        elif mode == "eps-greedy":
            if random.uniform(0, 1) < self.epsilon:
                if "action_dist" in self.config:
                    return np.expand_dims(
                        np.random.choice(a, size=n, p=self.config["action_dist"]), 1
                    )
                else:
                    return np.expand_dims(np.random.choice(a, size=n), 1)
            else:
                return np.expand_dims(np.argmax(q, axis=1), 1)
        raise NotImplementedError(
            "Cannot sample directly from Q-values, maybe treat them as logits later?"
        )
        return tf.random.categorical(tf.math.log(q), 1).numpy()

    def pad_history(self, obs):
        if self.env_name == "snake":
            # pad history if insufficient
            if obs.shape[-1] < 3 * self.config["n_frames"]:
                padding = 3 * self.config["n_frames"] - obs.shape[-1]
                return tf.concat(
                    [tf.zeros(obs.shape[:-1] + (padding,), dtype=tf.float64), obs],
                    axis=-1,
                )
        return obs

    def preprocess(self, obs):
        if self.env_name == "snake":  # subtract background colour
            return np.array(obs.astype(np.float32)[::10, ::10] / 255.0) - np.array(
                [0.0, 1.0, 0.0]
            )
        elif self.env_name in ("tetris", "tetris-simple"):
            return np.array(obs.astype(np.float32) / 255.0)
        elif self.env_name == "taxi":
            return np.array([obs])
        # elif self.env_name == 'breakout':
        #    return np.array(obs.astype(np.float32)[::5,::5]/127.0-1.0)
        return obs

    def data_preprocess(self, samples):
        # TODO: refactor
        def flip(state, dim):
            return tf.reverse(state, dim)

        def mirror(state):
            return tf.transpose(state, perm=[1, 0, 2])

        def augment(state, action, reward, p, ss):
            rand = np.random.random((3,))
            if rand[0] > 0.5:
                state = flip(state, [0])
                ss = flip(ss, [0])
                if action % 2 == 0:
                    action = (action + 2) % 4  # hack that only works for snake
            if rand[1] > 0.5:
                state = flip(state, [1])
                ss = flip(ss, [1])
                if action % 2 == 1:
                    action = 1 if action == 3 else 3
            if rand[2] > 0.5:
                state = mirror(state)
                ss = mirror(ss)
                if action in (0, 3):
                    action = 3 if action == 0 else 0
                else:
                    action = 2 if action == 1 else 1
            return Transition(state, action, reward, ss, p)

        if self.env_name == "snake":
            return [
                augment(s.state, s.action, s.reward, s.discount, s.next)
                for s in samples
            ]
        else:
            return samples

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

    def collect_rollout(
        self,
        t_max=10000,
        policy=None,
        silenced=True,
        train=False,
        display=False,
        add_buffer=True,
        eval=False,
    ) -> Union[float, Tuple[float, List]]:
        """Collects a rollout of experience with maximum length t_max, and returns the (undiscounted) reward"""
        if silenced:
            self.detach_stdout()

        obs = self.preprocess(self.env.reset())
        if display:
            self.env.render()
        dn = False
        i = 0
        reward = 0
        g = 1.0
        rt = 0
        queue = []
        state_queue = [obs]
        obs = self.pad_history(tf.concat(state_queue, axis=-1))
        if self.noisy_DQN or eval:
            action_mode = "greedy"
        else:
            action_mode = "eps-greedy"
        while i != t_max:
            act = (
                self.get_action(obs, mode=action_mode)[0][0]
                if policy is None
                else policy(obs)
            )
            oo, rr, dn, info = self.env.step(self.action_wrapper(act))
            if display:
                self.env.render()
            rt += g * rr
            oo = self.preprocess(oo)
            state_queue = (state_queue + [oo])[-self.config["n_frames"] :]
            oo = self.pad_history(tf.concat(state_queue, axis=-1))
            queue.append((obs, act, rr))
            # switch to using multistep returns
            # self.buffer.add((obs, act, rr, 0.0 if dn else self.gamma, oo))
            reward += rr
            obs = oo

            if len(queue) >= self.multistep:
                self.buffer.add(
                    Transition(
                        state=queue[0][0],
                        action=queue[0][1],
                        reward=rt,
                        discount=0.0 if dn else g * self.gamma,
                        next=oo,
                    )
                )
                rt -= queue[0][2]
                rt /= self.gamma
                del queue[0]
            else:
                g *= self.gamma

            # (TODO): arbitrarily chosen for now
            if train and (i + 1) % self.update_freq == 0:
                for _ in range(self.update_steps):
                    self.update_model_step()

            if dn or i + 1 == t_max:
                # clean up incomplete transitions in the queue
                while len(queue) > 0:
                    self.buffer.add(
                        Transition(
                            state=queue[0][0],
                            action=queue[0][1],
                            reward=rt,
                            discount=0.0,
                            next=oo,
                        )
                    )
                    rt /= self.gamma
                    rt -= queue[0][2]
                    del queue[0]
                break
            i += 1
        if silenced:
            self.attach_stdout()
        return reward

    def discount_rewards(self, r):
        for i in range(len(r) - 1, -1, -1):
            r[i] += 0 if i == len(r) - 1 else self.gamma * r[i + 1]
        return r

    def unpack(self, samples: List[Transition]):
        augmented = self.data_preprocess(samples)
        s, a, r, p, ss = [], [], [], [], []
        for sample in augmented:
            s.append(sample.state)
            a.append(sample.action)
            r.append(sample.reward)
            p.append(sample.discount)
            ss.append(sample.next)
        return np.array(s), np.array(a), np.array(r), np.array(p), np.array(ss)

    def compute_model_loss(self, s, a, r, p, ss, w=None):
        if self.double_DQN:
            idxs = np.argmax(self.get_model(ss, batch=True), axis=1)
            q = self.get_target(ss, batch=True)
            idx = tf.one_hot(idxs, tf.shape(q)[1])
            yy = tf.reduce_sum(tf.math.multiply(q, idx), axis=1)
        else:
            q = self.get_target(ss, batch=True)
            yy = np.max(q, axis=1)
        y = r + tf.multiply(p, yy)
        preds = self.get_model(s, batch=True)
        a = tf.one_hot(a, tf.shape(preds)[1])
        q_pred = tf.reduce_sum(tf.multiply(preds, a), axis=1)
        delta = q_pred - y
        losses = tf.square(delta)
        if self.prioritized_sampling:
            losses = losses * w
        loss = tf.reduce_mean(losses)

        if self.update_counter % 10 == 1:
            # print(q_pred, y)
            print(q_pred, r, yy)

        reg_loss = self.config["lambda"] * l2_loss(self.model)
        loss = loss + reg_loss
        return (loss, tf.abs(delta)) if self.prioritized_sampling else loss

    def update_model_step(self):
        if not self.training:
            return
        self.update_counter += 1
        if self.prioritized_sampling:
            idxs, w, samples = self.buffer.sample(
                self.batch_size  # if self.update_counter % 1000 == 0
                # else self.batch_size
            )
            s, a, r, p, ss = self.unpack(samples)
            with tf.GradientTape() as tape:
                model_loss, delta = self.compute_model_loss(
                    s,
                    a,
                    tf.constant(r, dtype=tf.float32),
                    tf.constant(p, dtype=tf.float32),
                    ss,
                    w,
                )
            model_grads = tape.gradient(model_loss, self.model.trainable_variables)
            clipped_grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in model_grads]
            self.optimizer.apply_gradients(
                zip(clipped_grads, self.model.trainable_variables)
            )
            delta = np.power(delta.numpy(), self.alpha)
            self.buffer.refresh_priority(idxs, delta)
        else:
            s, a, r, p, ss = self.unpack(self.buffer.sample(self.batch_size))
            with tf.GradientTape() as tape:
                model_loss = self.compute_model_loss(
                    s,
                    a,
                    tf.constant(r, dtype=tf.float32),
                    tf.constant(p, dtype=tf.float32),
                    ss,
                )
            model_grads = tape.gradient(model_loss, self.model.trainable_variables)
            clipped_grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in model_grads]
            self.optimizer.apply_gradients(
                zip(clipped_grads, self.model.trainable_variables)
            )

        for callback in self.callbacks:
            callback.on_train_step_end(self)

        self.update_target_step()

    def update_target_step(self):
        if (self.update_counter + 1) % self.target_delay == 0:
            model_weights = self.model.get_weights()
            target_weights = self.target.get_weights()
            self.target.set_weights(
                list(
                    map(
                        lambda x: self.delta * x[0] + (1.0 - self.delta) * x[1],
                        zip(model_weights, target_weights),
                    )
                )
            )

    # @tf.function(experimental_relax_shapes=True)
    def update_network(self, steps):
        for _ in range(steps):
            self.update_model_step()
        self.update_target_step()

    def train(self, epochs=None, t_max=None, logging=True, display=False):
        if self.update_counter == 0:  # hack to call init if not done so already
            for callback in self.callbacks:
                callback.on_init(self)
        if epochs is None:
            epochs = self.config.get("train_epochs", 1000)
        if t_max is None:
            t_max = self.config.get("t_max", 10000)
        avg_reward = 0.0
        epochs_per_log = self.config.get("log_interval", min(25, epochs / 10))
        hist = []
        for t in tqdm(range(epochs), desc="Training epochs"):
            reward = self.collect_rollout(
                t_max=t_max, silenced=True, train=True, display=display
            )
            avg_reward += reward
            hist.append(reward)

            if t % epochs_per_log == epochs_per_log - 1:
                avg_reward /= epochs_per_log
                if logging:
                    logger.info("[{}] Average reward: {}".format(t + 1, avg_reward))
                    logger.info(
                        "Evaluation reward: {}".format(
                            self.collect_rollout(
                                t_max=t_max,
                                silenced=True,
                                train=False,
                                eval=True,
                                display=display,
                            )
                        )
                    )
                    logger.log(
                        "Predicted reward: {}".format(
                            self.get_model(
                                self.pad_history(self.preprocess(self.env.reset()))
                            )
                        )
                    )
                    logger.log("Buffer size: {}".format(self.buffer.size()))
                avg_reward = 0
                self.save_to_checkpoint(logging, manual=False)
                self.load_from_checkpoint(manual=False)

            for callback in self.callbacks:
                callback.on_episode_end(self)

        return hist

    def load(self, path):
        model_path = os.path.join(path, "model.h5")
        self.model = tf.keras.models.load_model(model_path)

    def load_from_checkpoint(self, manual=False):
        if not manual:
            if len(self.ckpt_manager.checkpoints) > 0:
                logger.info(
                    f"Found existing checkpoint {self.ckpt_manager.latest_checkpoint}"
                )
                self.existing = True
            status = self.checkpoint().restore(self.ckpt_manager.latest_checkpoint)
            status.expect_partial()
        else:
            try:
                self.model.load_weights(os.path.join(self.ckpt_dir, "model_checkpoint"))
                self.target.load_weights(
                    os.path.join(self.ckpt_dir, "target_checkpoint")
                )
                self.existing = True
            except:
                logger.warning("Failed to load from checkpoint")

    def save(self, path=""):
        if len(path) == 0:
            now = datetime.now()
            path = "{}{}-{}".format(
                self.algo_name,
                "-{}".format(self.env_name) if len(self.env_name) > 0 else "",
                now.strftime("%H-%M-%S"),
            )
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "model.h5")
        self.model.save(model_path)

    def save_to_checkpoint(self, logging=True, manual=False):
        if not self.training:
            return
        if logging:
            logger.log("Saving to checkpoint...")

        if manual:
            self.model.save_weights(os.path.join(self.ckpt_dir, "model_checkpoint"))
            self.target.save_weights(os.path.join(self.ckpt_dir, "target_checkpoint"))

        assert self.ckpt_manager is not None
        self.ckpt_manager.save()
