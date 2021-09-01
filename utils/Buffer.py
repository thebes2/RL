import numpy as np

class ReplayBuffer:

    def __init__(self, max_size=1000000, num_samples=128, mode='uniform'):
        self.max_size = max_size
        self.num_samples = num_samples
        self.queue = []
        self.mode = mode
        if self.mode not in ('uniform', 'rank', 'proportional'):
            raise Exception("Invalid mode: {}".format(self.mode))

    def add(self, obs):
        if len(self.queue) >= self.max_size:
            del self.queue[0]
        self.queue.append(obs)

    def sample(self, num_samples=None):
        if num_samples is None:
            num_samples = self.num_samples
        if num_samples > self.size():
            num_samples = self.size()
        if self.mode == 'uniform':
            idxs = np.random.choice(len(self.queue), size=num_samples, replace=False)
            return list(map(lambda x: self.queue[x], idxs))
        else:
            raise NotImplementedError()

    def size(self):
        if self.mode == 'uniform':
            return len(self.queue)
        raise NotImplementedError()