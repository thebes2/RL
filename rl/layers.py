import tensorflow as tf


def apply_dense_along_axis(
    inp, axis: int, units: int, feature_dim: int, residual: bool = False
):
    """Apply dense layer along axis of input

    Args:
        inp: tensor of shape [B, H, W, D]
        axis: int in (1, 2) - the axis to apply the dense layer along
        units: number of output units
        feature_dim: feature dimension of
        residual: whether to output result or add it to input
    """
    assert len(inp.shape) == 4
    slices = tf.unstack(inp, axis=axis)
    regressor = tf.keras.layers.Dense(
        units * feature_dim, activation="relu" if not residual else None
    )
    batch_size = tf.shape(inp)[0]
    outputs = tf.stack(
        [
            tf.reshape(
                regressor(tf.keras.layers.Flatten()(slc)),
                [batch_size, units, feature_dim],
            )
            for slc in slices
        ],
        axis=axis,
    )
    if residual:
        # assert outputs.shape == inp.shape, f"{outputs.shape} must match {inp.shape}"
        outputs = tf.keras.layers.ReLU()(outputs + inp)
    return outputs


def get_normed_resblock(inp, dim: int, inner_dim: int):
    """Two-layer normed resblock"""
    hidden = tf.keras.layers.Dense(inner_dim, activation=None)(inp)
    bn1 = tf.keras.layers.BatchNormalization()(hidden)
    out1 = tf.keras.layers.Activation("relu")(bn1)
    hidden2 = tf.keras.layers.Dense(dim, activation=None)(out1)
    bn2 = tf.keras.layers.BatchNormalization()(hidden2)
    res = bn2 + inp
    out = tf.keras.layers.Activation("relu")(res)
    return out
