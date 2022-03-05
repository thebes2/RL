import numpy as np
import tensorflow as tf


def l2_loss(model: tf.keras.Model):
    return tf.add_n(
        [tf.nn.l2_loss(v) for v in model.trainable_variables if "bias" not in v.name]
    )


class Trainer:
    """
    A class for training a model
    Allows for a complex model like DQN to incorporate many training methods at once
    without complicating the main class
    """

    def __init__(self):
        pass

    def update(self, agent):
        pass


class DQNMSETrainer(Trainer):
    def __init__(self, log=False):
        super(DQNMSETrainer, self).__init__()
        self.prioritized_replay = False
        self.double_DQN = False
        self.log = log
        self.update_counter = 0

    def compute_loss(self, s, a, r, p, ss, w=None):
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
        if self.prioritized_replay:
            losses = losses * w
        loss = tf.reduce_mean(losses)
        if self.log and self.update_counter % 25 == 0:
            print("\n\n", q_pred, y, loss)
            print(tf.reduce_mean(tf.square(delta)))
        return (loss, tf.abs(delta)) if self.prioritized_replay else loss

    def compute_and_update_buffer(self, agent):
        if self.prioritized_replay:
            idxs, w, samples = agent.buffer.sample(self.batch_size)
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
            agent.buffer.refresh_priority(idxs, delta)
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

    def update(self, agent):
        self.update_counter += 1
        self.prioritized_replay = agent.prioritized_replay
        self.double_DQN = agent.double_DQN

        self.compute_and_update_buffer()
