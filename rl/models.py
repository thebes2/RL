import tensorflow as tf

from rl.layers import apply_dense_along_axis, get_normed_resblock


def get_policy_architecture(env_name, algo="PPO", head=None, tail=None, config=None):
    if env_name == "CartPole-v0":
        if "Dueling" in algo:
            inp = tf.keras.Input(shape=(4,))
            hidden1 = tf.keras.layers.Dense(16, activation="relu")(inp)
            hidden2 = tf.keras.layers.Dense(16, activation="relu")(hidden1)
            val = tf.keras.layers.Dense(1, activation="linear")(hidden2)
            adv = tf.keras.layers.Dense(2, activation="linear")(hidden2)
            avg = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, 1))(adv)
            out = tf.keras.layers.Add()([val, adv, -avg])
            model = tf.keras.Model(inputs=inp, outputs=out, name="dueling-DQN")
        else:
            model = tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(4,)),
                    tf.keras.layers.Dense(16, activation="relu"),
                    tf.keras.layers.Dense(16, activation="relu"),
                    tf.keras.layers.Dense(
                        2, activation="softmax" if algo == "PPO" else None
                    ),
                ]
            )
    elif env_name == "MountainCar-v0":
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(2,)),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(
                    3, activation="softmax" if algo == "PPO" else None
                ),
            ]
        )
    elif env_name == "Acrobot-v1":
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(6,)),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(48, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(
                    3, activation="softmax" if algo == "PPO" else None
                ),
            ]
        )
    elif env_name == "gym_snake:snake-v0":
        inp = tf.keras.Input(shape=(15, 15, 3 * config["n_frames"]))
        if head is None:
            head = get_vision_architecture(env_name, config=config)
        latent = head(inp)
        if tail is None:
            tail = get_qlearning_architecture(env_name, algo=algo)
        out = tail(latent)
        model = tf.keras.Model(inputs=inp, outputs=out, name="-".join(algo))
    elif env_name in ("tetris", "tetris-simple"):  # the final raid boss
        inp = tf.keras.Input(shape=(20, 10, 3 * config["n_frames"]))
        if head is None:
            head = get_vision_architecture(env_name, config=config)
        latent = head(inp)
        if tail is None:
            tail = get_qlearning_architecture(env_name, algo=algo)
        out = tail(latent)
        model = tf.keras.Model(inputs=inp, outputs=out, name="-".join(algo))
    elif env_name == "Breakout-v0":
        model = tf.keras.Sequential(
            [  # suggested model for breakout from tf
                tf.keras.Input(
                    shape=(
                        210,
                        160,
                        3,
                    )
                ),
                tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu"),
                tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu"),
                tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(4, activation=None),
            ]
        )
    elif env_name == "LunarLander-v2":
        if "Dueling" in algo:
            inp = tf.keras.Input(shape=(8,))
            hidden1 = tf.keras.layers.Dense(128, activation="relu")(inp)
            hidden2 = tf.keras.layers.Dense(128, activation="relu")(hidden1)
            hidden3 = tf.keras.layers.Dense(64, activation="relu")(hidden2)
            val = tf.keras.layers.Dense(1, activation="linear")(hidden3)
            adv = tf.keras.layers.Dense(4, activation="linear")(hidden3)
            avg = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, 1))(adv)
            out = tf.keras.layers.Add()([val, adv, -avg])
            model = tf.keras.Model(inputs=inp, outputs=out, name="dueling-DQN")
        else:
            model = tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(8,)),
                    tf.keras.layers.Dense(128, activation="relu"),
                    tf.keras.layers.Dense(128, activation="relu"),
                    tf.keras.layers.Dense(64, activation="relu"),
                    tf.keras.layers.Dense(
                        4, activation="softmax" if "PPO" in algo else None
                    ),
                ]
            )
    return model


def get_value_architecture(env_name):
    if env_name == "CartPole-v0":
        value = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(4,)),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1, activation=None),
            ]
        )
    elif env_name == "MountainCar-v0":
        raise NotImplementedError()
    elif env_name == "Acrobot-v1":
        value = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(6,)),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(48, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1, activation=None),
            ]
        )
    elif env_name == "gym_snake:snake-v0":
        value = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(15, 15, 3)),
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                # tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation=None),
            ]
        )
    elif env_name == "tetris":  # the final raid boss
        value = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(20, 10, 3)),
                tf.keras.layers.Conv2D(32, (3, 3), activation="elu"),
                tf.keras.layers.Conv2D(
                    64, (3, 3), activation="elu", padding="valid"
                ),  # new addition
                tf.keras.layers.Conv2D(128, (3, 3), activation="elu"),
                tf.keras.layers.Flatten(),
                # tf.keras.layers.Dense(1024, activation='relu'),
                tf.keras.layers.Dense(256, activation="elu"),
                tf.keras.layers.Dense(64, activation="elu"),
                tf.keras.layers.Dense(1, activation=None),
            ]
        )
    elif env_name == "LunarLander-v2":
        value = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(8,)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1, activation=None),
            ]
        )
    return value


SNAKE_EMBED_DIM = 64
TETRIS_EMBED_DIM = 256  # 1024
TETRIS_FEATURE_DIM = 96


def get_vision_architecture(env_name, config=None):

    if env_name == "gym_snake:snake-v0":
        inp = tf.keras.Input(shape=(15, 15, 3 * config["n_frames"]))
        ft = tf.keras.layers.Conv2D(4, (1, 1), activation=None)(inp)
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(
            ft
        )
        mp_col1 = apply_dense_along_axis(conv1, 2, 15, 32, residual=True)
        mp_row1 = apply_dense_along_axis(mp_col1, 1, 15, 32, residual=True)
        conv2 = tf.keras.layers.Conv2D(64, (5, 5), activation="relu", padding="same")(
            mp_row1
        )
        local_features = tf.keras.layers.Flatten()(conv1)
        large_features = tf.keras.layers.Flatten()(conv2)
        features = tf.keras.layers.concatenate([local_features, large_features])
        fc1 = tf.keras.layers.Dense(SNAKE_EMBED_DIM, activation="relu")(features)
        model = tf.keras.Model(inputs=inp, outputs=fc1, name="vision")
        # inp = tf.keras.Input(shape=(15, 15, 3 * config["n_frames"]))
        # ft = tf.keras.layers.Conv2D(4, (1, 1), activation=None)(inp)
        # conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(
        #     ft
        # )
        # conv2 = tf.keras.layers.Conv2D(64, (5, 5), activation="relu", padding="same")(
        #     conv1
        # )
        # local_features = tf.keras.layers.Flatten()(conv1)
        # large_features = tf.keras.layers.Flatten()(conv2)
        # features = tf.keras.layers.concatenate([local_features, large_features])
        # fc1 = tf.keras.layers.Dense(SNAKE_EMBED_DIM, activation="relu")(features)
        # model = tf.keras.Model(inputs=inp, outputs=fc1, name="vision")

    elif env_name in ("tetris", "tetris-simple"):
        inp = tf.keras.Input(shape=(20, 10, 3))
        ft = tf.keras.layers.Conv2D(4, (1, 1), activation=None)(inp)
        conv1 = tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", padding="same"  # 64
        )(ft)
        conv2 = tf.keras.layers.Conv2D(
            32, (5, 5), activation="relu", padding="same"  # 128, (5, 5)
        )(conv1)
        conv3 = tf.keras.layers.Conv2D(
            32, (5, 5), activation="relu", padding="same"  # 128
        )(conv2)
        small_features = tf.keras.layers.Flatten()(conv1)
        medium_features = tf.keras.layers.Flatten()(conv2)
        large_features = tf.keras.layers.Flatten()(conv3)
        features = tf.keras.layers.concatenate(
            [small_features, medium_features, large_features]
        )
        fc1 = tf.keras.layers.Dense(TETRIS_EMBED_DIM, activation="relu")(features)
        # features = tf.keras.layers.Concatenate(axis=-1)(
        #     [conv1, conv3]
        # )  # feature-dim = 96
        model = tf.keras.Model(inputs=inp, outputs=fc1, name="vision")

    else:
        raise AssertionError(
            "Vision architecture for {} not specified.".format(env_name)
        )

    return model


def get_qlearning_architecture(env_name, algo=None):

    if env_name == "gym_snake:snake-v0":
        inp = tf.keras.Input(shape=(SNAKE_EMBED_DIM,))
        if "Dueling" in algo:
            hidden = tf.keras.layers.Dense(32, activation="relu")(inp)
            hidden2 = tf.keras.layers.Dense(32, activation="relu")(hidden)
            val = tf.keras.layers.Dense(1, activation="linear")(hidden2)
            adv = tf.keras.layers.Dense(4, activation="linear")(hidden2)
            avg = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, 1))(adv)
            out = tf.keras.layers.Add()([val, adv, -avg])
        else:
            hidden = tf.keras.layers.Dense(64, activation="relu")(inp)
            out = tf.keras.layers.Dense(
                4, activation="softmax" if "PPO" in algo else None
            )(hidden)
        model = tf.keras.Model(inputs=inp, outputs=out, name="qlearning-head")

    elif env_name == "tetris":
        inp = tf.keras.Input(shape=(TETRIS_EMBED_DIM,))
        # inp = tf.keras.Input(shape=(20, 10, TETRIS_FEATURE_DIM))
        if "Dueling" in algo:
            hidden = get_normed_resblock(inp, TETRIS_EMBED_DIM, TETRIS_EMBED_DIM)
            hidden2 = get_normed_resblock(hidden, TETRIS_EMBED_DIM, TETRIS_EMBED_DIM)
            hidden3 = get_normed_resblock(hidden2, TETRIS_EMBED_DIM, TETRIS_EMBED_DIM)
            hidden4 = tf.keras.layers.Dense(32, activation="relu")(
                hidden3
            )  # get_normed_resblock(inp, TETRIS_EMBED_DIM, TETRIS_EMBED_DIM)(hidden2)
            # hidden = tf.keras.layers.Dense(128, activation="relu")(inp)  # 256
            # hidden2 = tf.keras.layers.Dense(32, activation="relu")(hidden)  # 128
            # hidden3 = tf.keras.layers.Dense(32, activation="relu")(hidden2)  # 32
            val = tf.keras.layers.Dense(1, activation="linear")(hidden4)  # hidden3
            adv = tf.keras.layers.Dense(7, activation="linear")(hidden4)  # hidden3
            avg = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, 1))(adv)
            out = tf.keras.layers.Add()([val, adv, -avg])
        else:
            raise NotImplementedError()
            hidden = tf.keras.layers.Dense(64, activation="elu")(inp)
            out = tf.keras.layers.Dense(
                7, activation="softmax" if "PPO" in algo else None
            )(hidden)

        model = tf.keras.Model(inputs=inp, outputs=out, name="qlearning-head")

    elif env_name == "tetris-simple":  # the mini raid boss
        inp = tf.keras.Input(shape=(TETRIS_EMBED_DIM,))
        # inp = tf.keras.Input(shape=(20, 10, TETRIS_FEATURE_DIM))
        if "Dueling" in algo:
            hidden = get_normed_resblock(inp, TETRIS_EMBED_DIM, TETRIS_EMBED_DIM)
            hidden2 = get_normed_resblock(hidden, TETRIS_EMBED_DIM, TETRIS_EMBED_DIM)
            # hidden3 = get_normed_resblock(hidden2, TETRIS_EMBED_DIM, TETRIS_EMBED_DIM)
            hidden3 = tf.keras.layers.Dense(64, activation="relu")(hidden2)
            val = tf.keras.layers.Dense(1, activation="linear")(hidden3)
            adv = tf.keras.layers.Dense(15, activation="linear")(hidden3)
            avg = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, 1))(adv)
            out = tf.keras.layers.Add()([val, adv, -avg])
        else:
            raise NotImplementedError()
            hidden = tf.keras.layers.Dense(64, activation="elu")(inp)
            out = tf.keras.layers.Dense(
                7, activation="softmax" if "PPO" in algo else None
            )(hidden)

        model = tf.keras.Model(inputs=inp, outputs=out, name="qlearning-head")

    return model


def get_transition_architecture(env_name, algo=None, cfg={}):

    if env_name == "gym_snake:snake-v0":
        inp = tf.keras.Input(shape=(SNAKE_EMBED_DIM + cfg["n_actions"],))
        proj = tf.keras.layers.Dense(SNAKE_EMBED_DIM, activation="relu")(inp)
        hidden1 = get_normed_resblock(proj, SNAKE_EMBED_DIM, 2 * SNAKE_EMBED_DIM)
        hidden2 = get_normed_resblock(hidden1, SNAKE_EMBED_DIM, 2 * SNAKE_EMBED_DIM)
        hidden3 = get_normed_resblock(hidden2, SNAKE_EMBED_DIM, 2 * SNAKE_EMBED_DIM)
        out = tf.keras.layers.Activation("relu")(hidden3)
        # hidden1 = tf.keras.layers.Dense(SNAKE_EMBED_DIM)(inp)
        # bn = tf.keras.layers.BatchNormalization()(hidden1)
        # act1 = tf.keras.layers.Activation("relu")(bn)
        # hidden2 = tf.keras.layers.Dense(SNAKE_EMBED_DIM, activation="relu")(act1)
        model = tf.keras.Model(inputs=inp, outputs=out, name="transition")

    elif env_name == "tetris":
        inp = tf.keras.Input(shape=(TETRIS_EMBED_DIM + cfg["n_actions"],))
        hidden1 = tf.keras.layers.Dense(TETRIS_EMBED_DIM)(inp)
        bn = tf.keras.layers.BatchNormalization()(hidden1)
        act1 = tf.keras.layers.Activation("relu")(bn)
        hidden2 = tf.keras.layers.Dense(TETRIS_EMBED_DIM, activation="relu")(act1)
        model = tf.keras.Model(inputs=inp, outputs=hidden2, name="transition")

    return model


def get_projection_architecture(env_name, algo=None, cfg={}):

    if env_name == "gym_snake:snake-v0":
        inp = tf.keras.Input(shape=(SNAKE_EMBED_DIM,))
        proj = tf.keras.layers.Dense(cfg["latent_dim"], activation=None)(inp)
        model = tf.keras.Model(inputs=inp, outputs=proj, name="projection")

    elif env_name == "tetris":
        inp = tf.keras.Input(shape=(TETRIS_EMBED_DIM,))
        proj = tf.keras.layers.Dense(cfg["latent_dim"], activation=None)(inp)
        model = tf.keras.Model(inputs=inp, outputs=proj, name="projection")

    return model


def get_prediction_architecture(env_name, cfg={}):

    if env_name == "gym_snake:snake-v0":
        inp = tf.keras.Input(shape=(cfg["latent_dim"],))
        proj = tf.keras.layers.Dense(cfg["latent_dim"], activation=None)(inp)
        model = tf.keras.Model(inputs=inp, outputs=proj, name="prediction")

    elif env_name == "tetris":
        inp = tf.keras.Input(shape=(cfg["latent_dim"],))
        proj = tf.keras.layers.Dense(cfg["latent_dim"], activation=None)(inp)
        model = tf.keras.Model(inputs=inp, outputs=proj, name="prediction")

    return model
