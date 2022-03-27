from typing import List

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from rl.models import (
    get_normed_resblock,
    get_policy_architecture,
    get_prediction_architecture,
    get_projection_architecture,
    get_transition_architecture,
    get_vision_architecture,
)
from utils.Buffer import ReplayBuffer, Transition, is_deterministic
from utils.Conv import ConvHead
from utils.logger import logger


class Callback:
    """
    Callbacks expose hooks to the training process and allow modular/independent
    behaviors without modifying the main model
    """

    def __init__(self):
        pass

    def on_init(self, agent):
        pass

    def on_episode_end(self, agent):
        pass

    def on_train_step_end(self, agent):
        pass


def get_callback_or_value(x):
    if type(x) is dict and "type" in x:
        proc_args = {k: get_callback_or_value(v) for k, v in x["kwargs"].items()}
        return globals()[x["type"]](**proc_args)
    elif type(x) is list:
        return get_callbacks(x)
    else:
        return x


def get_callbacks(config):
    """Translates list of callback specifications into a list of actual callbacks"""
    return list(map(get_callback_or_value, config))


class Schedule:
    def __init__(self, length, start_val, end_val=0, fn="constant"):
        if fn not in ("constant", "linear"):
            raise NotImplementedError(
                "Only constant and linear scaling is supported for now"
            )
        self.length = length
        self.start_val = start_val
        self.end_val = end_val
        self.fn = fn

    def get_val(self, cnt):
        if self.fn == "constant":
            return self.start_val
        elif self.fn == "linear":
            return (self.end_val - self.start_val) / self.length * cnt + self.start_val


class AnnealingSchedulerCallback(Callback):
    def __init__(self, target, schedule):
        super(AnnealingSchedulerCallback, self).__init__()
        self.counter = 0
        self.target = target
        self.schedule = schedule
        self.last_val = 0

    # executed immediately before training begins
    def on_init(self, agent):
        self._set_val(self.counter, agent)

    def _set_val(self, cnt, agent):
        val = None
        for x in self.schedule:
            if cnt < x.length:
                val = x.get_val(cnt)
                break
            cnt -= x.length
        if val is None:
            val = self.last_val
        self.last_val = val
        setattr(agent, self.target, val)

    def on_episode_end(self, agent):
        self._set_val(self.counter, agent)
        self.counter += 1


class PretrainCallback(Callback):

    """
    Used for training the convolutional head for models that operate
    on raw pixels rather than vectors

    Currently supports using AM-Softmax to generate embeddings for
    inputs from the environment rather than using reconstruction loss

    Currently supports using a policy that samples actions uniformly
    at random.
    """

    def __init__(
        self,
        episodes=100,
        train_epochs=5,
        embed_dim=16,
        learning_rate=3e-4,
        policy="random",
    ):
        super(PretrainCallback, self).__init__()
        self.episodes = episodes
        self.train_epochs = train_epochs
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        if policy not in ("random", "greedy", "eps-greedy"):
            raise NotImplementedError("Invalid policy " + policy)
        self.policy = policy

    def on_init(self, agent):
        if agent.existing or not agent.training:
            return
        head = get_vision_architecture(agent.raw_env_name)
        embed = tf.keras.layers.Dense(self.embed_dim, activation=None)(head.output)
        model = tf.keras.Model(inputs=head.input, outputs=embed)

        p_buf = []

        def collect_rollout(env, t_max, policy):
            s = agent.preprocess(env.reset())
            for _ in range(t_max):
                act = policy(s)
                ss, r, dn, _ = env.step(agent.action_wrapper(act))
                ss = agent.preprocess(ss)
                if agent.env_name == "snake" and r > 0:
                    pass
                else:
                    p_buf.append([s, ss])
                s = ss
                if dn:
                    break

        if self.policy == "random":

            def policy(x):
                return np.random.choice(agent.n_actions)

        elif self.policy == "eps-greedy":

            def policy(x):
                return agent.get_action(x)

        elif self.policy == "greedy":

            def policy(x):
                return agent.get_action(x, mode="greedy")

        for _ in tqdm(range(self.episodes)):
            collect_rollout(agent.env, agent.t_max, policy)

        trainer = ConvHead(model, p_buf, lr=self.learning_rate)
        trainer.train(self.train_epochs)

        features = trainer.model.layers[-2].output
        trained_model = tf.keras.Model(inputs=trainer.model.input, outputs=features)
        agent.set_model(
            get_policy_architecture(agent.raw_env_name, agent.algo, trained_model)
        )


class SPRPretrainCallback(Callback):
    def __init__(
        self,
        episodes=100,
        train_epochs=5,
        learning_rate=3e-4,
        delta=0.0001,
        pred_len=10,
        policy="random",
    ):
        super(SPRPretrainCallback, self).__init__()
        self.episodes = episodes
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate
        self.policy = policy
        if policy not in ("random", "greedy", "eps-greedy"):
            raise NotImplementedError("Invalid policy " + policy)
        self.delta = delta
        self.pred_len = pred_len
        self.config = dict()

    def compute_loss(
        self,
        vision,
        vision_target,
        transition,
        projection,
        projection_target,
        prediction,
        trajectories: List[List[Transition]],
    ):  # TODO: code is somewhat duplicated from SPR_agent here
        def is_nan(t):
            return tf.math.reduce_any(tf.math.is_nan(t))

        inp = tf.stack([trajectories[i][0].state for i in range(len(trajectories))])

        z_0 = vision(inp)
        z_t = vision_target(inp)
        y_pred = prediction(projection(z_0))
        y_target = projection_target(z_t)
        y_pred = y_pred / tf.expand_dims(
            tf.sqrt(tf.reduce_sum(tf.square(y_pred), 1)), -1
        )
        y_target = y_target / tf.expand_dims(
            tf.sqrt(tf.reduce_sum(tf.square(y_target), 1)), -1
        )
        loss = -tf.reduce_sum(tf.linalg.matmul(y_pred, y_target, transpose_b=True))

        trajectories.append([])
        n = len(trajectories)
        lst = [
            next(j for j in range(n) if len(trajectories[j]) <= i)
            for i in range(len(trajectories[0]))
        ]

        def cond(i, _, __):
            return i < len(trajectories[0])

        eps = 1e-6

        def step(i, z, loss):
            l = lst[i]
            s = [traj[i] for traj in trajectories[:l]]
            act = [traj[i - 1].action for traj in trajectories[:l]]
            oh = np.eye(self.config["n_actions"])[act]
            inp = np.concatenate((z[:l], oh), axis=1)
            z_pred = transition(inp)
            z_target = vision_target(np.stack(map(lambda x: x.state, s)))
            y_pred = prediction(projection(z_pred))
            y_target = projection_target(z_target)
            y_pred = y_pred / tf.expand_dims(
                tf.sqrt(tf.reduce_sum(tf.square(y_pred), 1)) + eps, -1
            )
            y_target = y_target / tf.expand_dims(
                tf.sqrt(tf.reduce_sum(tf.square(y_target), 1)) + eps, -1
            )
            return (
                i + 1,
                z_pred,
                loss - tf.reduce_sum(tf.multiply(y_pred, y_target)),
            )

        _, _, loss = tf.while_loop(cond, step, (1, z_0, loss))
        return loss / sum(lst) / self.config["latent_dim"]

    def average_weights(model, target, delta):
        model_weights = model.get_weights()
        target_weights = target.get_weights()
        target.set_weights(
            list(
                map(
                    lambda x: delta * x[0] + (1.0 - delta) * x[1],
                    zip(model_weights, target_weights),
                )
            )
        )

    def on_init(self, agent) -> None:
        if agent.existing or not agent.training:
            return

        self.config = agent.config

        buffer = ReplayBuffer(env=agent.env_name, mode="spr", steps=self.pred_len)
        vision = get_vision_architecture(agent.raw_env_name, algo=agent.algo)
        vision_target = get_vision_architecture(agent.raw_env_name)
        transition = get_transition_architecture(agent.raw_env_name, cfg=agent.config)
        projection = get_projection_architecture(agent.raw_env_name, cfg=agent.config)
        projection_target = get_projection_architecture(
            agent.raw_env_name, cfg=agent.config
        )
        prediction = get_prediction_architecture(agent.raw_env_name, cfg=agent.config)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, clipnorm=1.0
        )

        trainable_variables = sum(  # do not directly train targets
            (
                model.trainable_variables
                for model in (
                    vision,
                    transition,
                    projection,
                    prediction,
                )
            ),
            [],
        )

        def collect_rollout(env, t_max, policy):
            s = agent.preprocess(env.reset())
            for _ in range(t_max):
                act = policy(s)
                ss, r, dn, _ = env.step(agent.action_wrapper(act))
                ss = agent.preprocess(ss)
                buffer.add(Transition(s, act, r, ss, 0.0 if dn else agent.gamma))
                s = ss
                if dn:
                    break

        if self.policy == "random":

            def policy(x):
                return np.random.choice(agent.n_actions)

        elif self.policy == "eps-greedy":

            def policy(x):
                return agent.get_action(x)

        elif self.policy == "greedy":

            def policy(x):
                return agent.get_action(x, mode="greedy")

        for _ in tqdm(range(self.episodes), desc="Collecting rollouts"):
            collect_rollout(agent.env, agent.t_max, policy)

        for _ in (
            pbar := tqdm(
                range(buffer.size() // buffer.num_samples * self.train_epochs),
                desc="Training SPR",
            )
        ):
            samples = buffer.sample()
            samples = list(sorted(samples, key=lambda l: len(l), reverse=True))
            with tf.GradientTape() as tape:
                loss = self.compute_loss(
                    vision,
                    vision_target,
                    transition,
                    projection,
                    projection_target,
                    prediction,
                    samples,
                )
                reg_loss = 0.0001 * tf.add_n(
                    [
                        tf.nn.l2_loss(v)
                        for v in trainable_variables
                        if "bias" not in v.name
                    ]
                )
                loss = loss + reg_loss
            pbar.set_postfix({"Loss": float(loss.numpy())})
            grads = tape.gradient(loss, trainable_variables)
            clipped_grads = (
                grads  # [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
            )
            optimizer.apply_gradients(zip(clipped_grads, trainable_variables))

            SPRPretrainCallback.average_weights(vision, vision_target, self.delta)
            SPRPretrainCallback.average_weights(
                projection, projection_target, self.delta
            )

        agent.set_model(
            get_policy_architecture(agent.raw_env_name, agent.algo, head=vision)
        )


class TetrisPretrainCallback(Callback):

    """
    SSL for Tetris vision head
    """

    def __init__(
        self,
        samples=100,
        train_epochs=5,
        embed_dim=16,
        learning_rate=3e-4,
        batch_size=64,
        policy="random",
    ):
        super(TetrisPretrainCallback, self).__init__()
        self.samples = samples
        self.train_epochs = train_epochs
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        if policy not in ("random", "greedy", "eps-greedy"):
            raise NotImplementedError("Invalid policy " + policy)
        self.policy = policy

    def on_init(self, agent):
        # currently we will pretrain by predicting max column heights
        # TODO: add predicting piece position, rotation, etc. later
        # also predicting rewards between timesteps
        if agent.existing or not agent.training:
            return
        head = get_vision_architecture(agent.raw_env_name)
        downscale = tf.keras.layers.Dense(128, activation=None)(head.output)
        # res1 = get_normed_resblock(downscale, 128, 256)
        # res2 = get_normed_resblock(res1, 128, 256)
        # res3 = get_normed_resblock(res2, 128, 256)
        # res4 = get_normed_resblock(res3, 128, 256)
        # res5 = get_normed_resblock(res4, 128, 256)
        # res6 = get_normed_resblock(res5, 128, 256)
        out = tf.keras.layers.Dense(11, activation=None)(head.output)
        model = tf.keras.Model(inputs=head.input, outputs=out)

        p_buf = []

        def collect_rollout(env, t_max, policy):
            s = agent.preprocess(env.reset())
            for _ in range(t_max):
                act = policy(s)
                ss, r, dn, info = env.step(agent.action_wrapper(act))
                ss = agent.preprocess(ss)
                target = (
                    info["column_heights"]
                    + [info["max_column_height"]]
                    # + list(info["position"])
                )
                p_buf.append([s, target])
                s = ss
                if dn:
                    break

        if self.policy == "random":

            def policy(_):
                return np.random.choice(agent.n_actions)

        elif self.policy == "eps-greedy":

            def policy(x):
                return agent.get_action(x)

        elif self.policy == "greedy":

            def policy(x):
                return agent.get_action(x, mode="greedy")

        with tqdm(total=self.samples) as pbar:
            while len(p_buf) < self.samples:
                sz = len(p_buf)
                collect_rollout(
                    agent.env,
                    t_max=agent.t_max,
                    policy=policy,
                )
                pbar.update(len(p_buf) - sz)

        def compute_loss(pred, gt):
            assert pred.shape == gt.shape
            return tf.reduce_mean(tf.square(pred - tf.cast(gt, tf.float32)))

        trainer = ConvHead(
            model, p_buf, compute_loss, lr=self.learning_rate, contrastive=False
        )
        trainer.train(self.train_epochs)

        features = trainer.model.layers[-2].output
        trained_model = tf.keras.Model(inputs=trainer.model.input, outputs=features)
        agent.set_model(
            get_policy_architecture(
                agent.raw_env_name, agent.algo, head=trained_model, config=agent.config
            )
        )


class InitBufferCallback(Callback):

    """
    Fills replay buffer with some transitions from the environment
    under a random policy
    """

    def __init__(self, samples: int = 1000, policy: str = "random"):
        self.samples = samples
        self.policy = policy

        assert self.policy in ("random", "action_dist")

    def on_init(self, agent) -> None:
        if not agent.training:
            return
        if self.policy == "random":
            sampling_policy = lambda _: np.random.choice(agent.n_actions)
        elif (
            self.policy == "action_dist"
        ):  # fall back to the agent's sampling policy distribution
            sampling_policy = lambda _: np.random.choice(
                agent.n_actions, p=agent.config["action_dist"]
            )
        with tqdm(total=self.samples) as pbar:
            while agent.buffer.size() < self.samples:
                sz = agent.buffer.size()
                agent.collect_rollout(
                    t_max=agent.t_max,
                    policy=sampling_policy if not agent.existing else None,
                    train=False,
                    display=False,
                )
                pbar.update(agent.buffer.size() - sz)


if __name__ == "__main__":
    a = get_callbacks(
        [
            {
                "type": "AnnealingSchedulerCallback",
                "kwargs": {
                    "target": "epsilon",
                    "schedule": [
                        {
                            "type": "Schedule",
                            "kwargs": {
                                "length": 10,
                                "start_val": 0.3,
                                "end_val": 0.0,
                                "fn": "linear",
                            },
                        }
                    ],
                },
            }
        ]
    )

    class mock:
        def __init__(self, callbacks):
            self.epsilon = 0.10
            self.callbacks = callbacks

            for c in self.callbacks:
                c.on_init(self)

        def train(self, episodes):
            for t in range(episodes):
                print(self.epsilon)

                for c in self.callbacks:
                    c.on_episode_end(self)

    b = mock(a)
    b.train(15)
