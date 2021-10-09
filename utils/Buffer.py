import numpy as np

class SegmentTree:

    def __init__(self, N):
        self.N = N
        # each node stores sum and max of range
        self.seg = [(0, 0, 0)] * 4 * N # avoid division by 0

    def get_sum(self):
        return self.seg[1][0]

    def get_max(self):
        return self.seg[1][1]

    def upd(self, i, s, e, idx, val):
        if s != e:
            if (s+e)/2 < idx:
                self.upd(2*i+1, (s+e)//2+1, e, idx, val)
            else:
                self.upd(2*i, s, (s+e)//2, idx, val)
            self.seg[i] = (
                self.seg[2*i][0] + self.seg[2*i+1][0],
                max(self.seg[2*i][1], self.seg[2*i+1][1]),
                min(self.seg[2*i][2], self.seg[2*i+1][2])
            )
        else:
            self.seg[i] = (val, val, val)

    def update(self, idx, val):
        self.upd(1, 0, self.N-1, idx, val)

    def qu(self, i, s, e, x):
        while s != e:
            if self.seg[2*i][0] < x:
                i, s, x = 2*i+1, (s+e)//2+1, x-self.seg[2*i][0]
            else:
                i, e = 2*i, (s+e)//2
        return s

    def query(self, x):
        return self.qu(1, 0, self.N-1, x)

    def qu2(self, i, s, e, idx):
        while s != e:
            if (s+e)/2 < idx:
                i, s = 2*i+1, (s+e)//2+1
            else:
                i, e = 2*i, (s+e)//2
        return self.seg[i][0]

    def get(self, idx):
        return self.qu2(1, 0, self.N-1, idx)

    def qu3(self, i, s, e, ss, se):
        if s >= ss and e <= se:
            return self.seg[i][2]
        elif (s+e)/2 < ss:
            return self.qu3(2*i+1, (s+e)//2+1, e, ss, se)
        elif (s+e)/2 >= se:
            return self.qu3(2*i, s, (s+e)//2, ss, se)
        else:
            return min(
                self.qu3(2*i+1, (s+e)//2+1, e, ss, se),
                self.qu3(2*i, s, (s+e)//2, ss, se)
            )

    def get_min(self, n):
        return self.qu3(1, 0, self.N-1, 0, n-1)


class ReplayBuffer:

    def __init__(self,
                 max_size=1000000,
                 num_samples=128,
                 samples_per_rebuild=10000,
                 beta=1.0,
                 mode='uniform'):
        self.max_size = max_size
        self.num_samples = num_samples
        self.mode = mode
        self.counter = 0
        self.samples = 0
        if self.mode not in ('uniform', 'rank', 'proportional'):
            raise Exception("Invalid mode: {}".format(self.mode))

        if self.mode == 'uniform':
            self.queue = []
        elif self.mode == 'proportional':
            self.arr = [None] * max_size
            self.index = 0
            self.samples_per_rebuild = samples_per_rebuild
            self.seg = SegmentTree(self.max_size)
            # (TODO): Add callback for annealing beta to 1.0 from beta_0
            self.beta = beta
            self.seg.update(0, 1e-1)
        else:
            raise NotImplementedError()

    def add(self, obs):
        self.counter += 1
        if self.mode == 'uniform':
            if len(self.queue) >= self.max_size:
                del self.queue[0]
            self.queue.append(obs)
        elif self.mode == 'proportional':
            self.arr[self.index] = obs
            self.seg.update(self.index, self.seg.get_max())
            self.index += 1
            if self.index == self.max_size:
                self.index = 0

    def sample(self, num_samples=None):
        self.samples += 1
        if num_samples is None:
            num_samples = self.num_samples
        if num_samples > self.size():
            num_samples = self.size()
        if self.mode == 'uniform':
            idxs = np.random.choice(len(self.queue), size=num_samples, replace=False)
            return list(map(lambda x: self.queue[x], idxs))
        elif self.mode == 'proportional':
            idxs = []
            # slightly different from PER paper
            for i in range(num_samples):
                idxs.append(self.seg.query(np.random.uniform(0, self.seg.get_sum())))
            # return IS weights for each sample as well
            probs = np.array(list(map(lambda x: self.seg.get(x), idxs)))
            weights = np.power(self.seg.get_sum() * np.reciprocal(probs) / self.size(), self.beta)
            maxw = np.power(self.seg.get_sum() / self.seg.get_min(self.size()) / self.size(), self.beta)
            # weights = weights / maxw
            return idxs, weights, list(map(lambda x: self.arr[x], idxs))
        else:
            raise NotImplementedError()

    def refresh_priority(self, idxs, prior):
        for idx, p in zip(idxs, prior):
            self.seg.update(idx, p)

    def size(self):
        if self.mode == 'uniform':
            return len(self.queue)
        elif self.mode == 'proportional':
            return min(self.counter, self.max_size)
        raise NotImplementedError()