import tensorflow as tf
import random
import numpy as np
from tqdm.auto import tqdm

def get_mask(y):
    indices_equal = tf.cast(tf.eye(tf.shape(y)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    labels_equal = tf.equal(tf.expand_dims(y, 0), tf.expand_dims(y, 1))
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return tf.cast(mask, tf.float32)


def am_softmax_loss(x, y, scale=20.0):
    l2 = tf.sqrt(tf.reduce_sum(tf.square(x), 1))
    x = x / tf.reshape(l2, (-1, 1)) # rescale

    x = tf.matmul(x, tf.transpose(x)) # pairwise similarities
    mask = get_mask(y)
    x = x - tf.eye(tf.shape(x)[0]) * 1000

    x = x * scale
    loss = tf.nn.softmax_cross_entropy_with_logits(mask, x)
    
    return loss


class Gen(tf.keras.utils.Sequence):

    def __init__(self, data, batch_size=128):
        self.data = data
        self.batch_size = batch_size//2

        self.on_epoch_end()

    def __len__(self):
        return len(self.data) // self.batch_size

    def on_epoch_end(self):
        random.shuffle(self.data)

    def __getitem__(self, idx):
        samples = self.data[idx*self.batch_size:(idx+1)*self.batch_size]
        X = np.zeros((2*len(samples),) + samples[0][0].shape)
        y = np.zeros(2*len(samples))
        for i, sample in enumerate(samples):
            X[2*i], X[2*i+1] = sample
            y[2*i], y[2*i+1] = (i, i)
        return X, y


class ConvHead:
    """
    Vision tower that forms the first few layers of the model.
    We train this like we do with image embeddings and hope that the features
    are useful for identifying the state and thus useful for estimating Q-values
    as well.
    """

    def __init__(self, architecture, data, lr=3e-4):
        self.model = architecture
        self.gen = Gen(data)

        self.optimizer = tf.optimizers.Adam(lr)

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            embed = self.model(x)
            loss = am_softmax_loss(embed, y)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss        

    def train(self, n_epochs=5):
        self.model.compile(
            optimizer=self.optimizer,
            loss=am_softmax_loss
        )

        for t in range(n_epochs):
            avg = 0
            for x,y in tqdm(self.gen):
                avg += self.train_step(x,y)
            # print("[{}] Loss: {}".format(t, avg/self.gen.__len__()))
            self.gen.on_epoch_end()

        #self.model.fit_generator(
        #    self.gen,
        #    epochs=n_epochs
        #)


if __name__ == '__main__':
    vecs = tf.constant([
        [1, 0],
        [0, 1],
        [3, 3],
        [3.5, 4]
    ])
    with tf.GradientTape() as tape:
        tape.watch(vecs)
        labels = tf.constant([0, 0, 1, 1])
        loss = am_softmax_loss(vecs, labels)
    grads = tape.gradient(loss, vecs)
    print(grads)