import tensorflow as tf

class Callback:

    def __init__(self):
        pass

    def on_episode_end(self, agent):
        pass

    def on_train_step_end(self, agent):
        pass


class Schedule:

    def __init__(self, length, start_val, end_val=0, fn='constant'):
        if fn not in ('constant', 'linear'):
            raise NotImplementedError("Only constant and linear scaling is supported for now")
        self.length = length
        self.start_val = start_val
        self.end_val = end_val
        self.fn = fn

    def get_val(self, cnt):
        if fn == 'constant':
            return self.start_val
        elif fn == 'linear':
            return (self.end_val - self.start_val) / self.length * cnt + self.start_val


class AnnealingSchedulerCallback(Callback):

    def __init__(self, target, schedule):
        super(AnnealingSchedulerCallback, self).__init__()
        self.counter = 0
        self.target = target
        self.schedule = schedule
        self._set_val(self.counter)

    def _set_val(self, cnt, agent):
        val = None
        for x in self.schedule:
            if cnt < x.length:
                val = x.get_val(cnt)
                break
            cnt -= x.length
        if val is None:
            val = self.schedule[-1].get_val(cnt)
        setattr(agent, self.target, val)

    def on_episode_end(self, agent):
        self._set_val(self.counter, agent)
        self.counter += 1