import tensorflow as tf

from rl.models import get_normed_resblock


def build_policy(config: dict):
    """
    Load model from config.
    Assumes the model is always composed of convolutions followed by fully connected
    """
    input_shape = config["input_shape"][:-1] + [
        config["input_shape"][-1] * config["n_frames"]
    ]

    _input = tf.keras.Input(shape=input_shape)
    _x = _input
    if "vision_cfg" in config:
        _x = build_vision(config, _x, input_shape)
    if "qlearning_cfg" in config:
        _x = build_qlearning(config, tf.shape(_x)[-1], _x)
    _outputs = _x

    return tf.keras.Model(
        inputs=_input, outputs=_outputs, name="-".join(config["algo"])
    )


def build_vision(config: dict, _input=None, input_shape=None):
    """
    Build vision head from config.
    Currently, this is always some form of PyramidNet
    """
    model_config = config["vision_cfg"]

    # since the envs we test on are very dense, we do not need stride for now
    assert "filters" in model_config
    assert "kernel_size" in model_config
    assert "downsample_factor" in model_config

    n_layers = len(model_config["filters"])
    assert n_layers > 0
    assert model_config["kernel_size"] == model_config["downsample_factor"] == n_layers

    if "keep_idxs" not in model_config:
        model_config["keep_idxs"] = range(n_layers)

    if _input is None:
        _input = tf.keras.Input(shape=input_shape)

    _x = _input
    features = []
    for filter, kernel, downsample, keep in zip(
        model_config["filters"],
        model_config["kernel_size"],
        model_config["downsample_factor"],
        model_config["keep_idxs"],
    ):
        _x = tf.keras.layers.Conv2D(filter, kernel, activation="relu", padding="same")(
            _x
        )
        if keep:
            features.append(_x)
        if downsample > 1:
            _x = tf.keras.layers.MaxPool2D(
                pool_size=(downsample, downsample), padding="same"
            )(_x)

    return tf.keras.layers.concatenate(
        [tf.keras.layers.Flatten()(feature) for feature in features]
    )


def build_qlearning(config: dict, input_dim: int, _input=None):
    """
    Build qlearning head from config.
    This is composed of a series of normed resblocks, followed by normal dense layers,
    and (optionally) by a dueling mechanism
    """
    model_config = config["qlearning_cfg"]

    if _input is None:
        _input = tf.keras.Input(shape=input_dim)

    _x = _input

    resblock_sizes = model_config.get("resblocks", [])

    for block in resblock_sizes:
        if input_dim != block:
            _x = tf.keras.layers.Dense(block, activation="relu")(_x)
        # for simplicity, we will use 2*outer dim as the inner input dimension
        _x = get_normed_resblock(_x, 2 * block, block)

    dense_sizes = model_config.get("dense_layers", [])

    for size in dense_sizes:
        _x = tf.keras.layers.Dense(size, activation="relu")(_x)

    if "Dueling" in config["algo"]:
        val = tf.keras.layers.Dense(1, activation="linear")(_x)
        adv = tf.keras.layers.Dense(config["n_actions"], activation="linear")(_x)
        avg = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, 1))(adv)
        _x = tf.keras.layers.Add()([val, adv, -avg])
    else:
        _x = tf.keras.layers.Dense(config["n_actions"], activation="linear")(_x)

    return _x
