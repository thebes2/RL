import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm

from rl.models import get_policy_architecture, get_vision_architecture
from utils.Conv import ConvHead

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
    if type(x) is dict and 'type' in x:
        proc_args = { k: get_callback_or_value(v) for k,v in x['kwargs'].items()}
        return globals()[x['type']](**proc_args)
    elif type(x) is list:
        return get_callbacks(x)
    else:
        return x


def get_callbacks(config):
    """Translates list of callback specifications into a list of actual callbacks"""
    return list(map(get_callback_or_value, config))


class Schedule:

    def __init__(self, length, start_val, end_val=0, fn='constant'):
        if fn not in ('constant', 'linear'):
            raise NotImplementedError("Only constant and linear scaling is supported for now")
        self.length = length
        self.start_val = start_val
        self.end_val = end_val
        self.fn = fn

    def get_val(self, cnt):
        if self.fn == 'constant':
            return self.start_val
        elif self.fn == 'linear':
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

    def __init__(self, 
                 episodes=100, 
                 train_epochs=5,
                 embed_dim=16,
                 learning_rate=3e-4,
                 policy='random'):
        super(PretrainCallback, self).__init__()
        self.episodes = episodes
        self.train_epochs = train_epochs
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        if policy not in ('random', 'greedy', 'eps-greedy'):
            raise NotImplementedError("Invalid policy " + policy)
        self.policy = policy

    def on_init(self, agent):
        if agent.existing:
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
                p_buf.append([s, ss])
                s = ss
                if dn:
                    break
        
        if self.policy == 'random':
            policy = lambda x: np.random.choice(agent.n_actions)
        elif self.policy == 'eps-greedy':
            policy = lambda x: agent.get_action(x)
        elif self.policy == 'greedy':
            policy = lambda x: agent.get_action(x, mode='greedy')
        for _ in tqdm(range(self.episodes)):
            collect_rollout(agent.env, agent.t_max, policy)
        
        trainer = ConvHead(model, p_buf, lr=self.learning_rate)
        trainer.train(self.train_epochs)

        features = trainer.model.layers[-2].output
        trained_model = tf.keras.Model(inputs=trainer.model.input, outputs=features)
        agent.set_model(get_policy_architecture(
            agent.raw_env_name,
            agent.algo,
            trained_model
        ))


class InitBufferCallback(Callback):

    """
        Fills replay buffer with some transitions from the environment
        under a random policy
    """

    def __init__(self, episodes=100):
        self.episodes = episodes

    def on_init(self, agent):
        for _ in range(self.episodes):
            agent.collect_rollout(
                t_max=agent.t_max,
                policy=lambda x: np.random.choice(agent.n_actions),
                train=False,
                display=False
            )


if __name__ == '__main__':
    a = get_callbacks([
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
                            "fn": "linear"
                        }
                    }
                ]
            }
        }
    ])

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