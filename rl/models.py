from collections import OrderedDict

import tensorflow as tf


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
TETRIS_FEATURE_DIM = 2 * 200 + 2 * 4
TETRIS_EMBED_DIM = TETRIS_FEATURE_DIM  # 1024


def get_vision_architecture(env_name, config=None):

    if env_name == "gym_snake:snake-v0":
        inp = tf.keras.Input(shape=(15, 15, 3 * config["n_frames"]))
        ft = tf.keras.layers.Conv2D(4, (1, 1), activation=None)(inp)
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(
            ft
        )
        conv2 = tf.keras.layers.Conv2D(64, (5, 5), activation="relu", padding="same")(
            conv1
        )
        local_features = tf.keras.layers.Flatten()(conv1)
        large_features = tf.keras.layers.Flatten()(conv2)
        features = tf.keras.layers.concatenate([local_features, large_features])
        fc1 = tf.keras.layers.Dense(SNAKE_EMBED_DIM, activation="relu")(features)
        model = tf.keras.Model(inputs=inp, outputs=fc1, name="vision")

    elif env_name == "tetris-simple":  # hack

        output_features = [
            "column_outputs",
            "column_global_features",
        ]  # which tensors to include in output
        concat = True  # flatten and concatenate features
        downscale = False  # downscale output feature vector to fixed dimension

        tensor_dict = OrderedDict()
        _input = tf.keras.Input(shape=(20, 10, 3))
        tensor_dict["input"] = _input
        # transform rgb pixel values to more usable format
        tensor_dict["transformed_input"] = tf.keras.layers.Conv2D(
            4, (1, 1), activation=None
        )(tensor_dict["input"])

        # convolutional backbone
        tensor_dict["conv1"] = tf.keras.layers.Conv2D(
            64, (5, 3), padding="same", activation="relu"
        )(tensor_dict["transformed_input"])

        # extract column features
        compact_columns = tf.keras.layers.Conv2D(4, (1, 1), activation=None)(
            tensor_dict["conv1"]
        )
        columns = tf.unstack(compact_columns, axis=2)  # 10 x [None x 20 x 4]
        flat_columns = [
            tf.keras.layers.Flatten()(column) for column in columns
        ]  # 10 x [None x 80]
        column_dense_layer1 = tf.keras.layers.Dense(40, activation="relu")
        column_dense_layer2 = tf.keras.layers.Dense(40, activation="relu")
        raw_column_outputs = [
            tf.expand_dims(column_dense_layer2(column_dense_layer1(col)), 1)
            for col in flat_columns
        ]
        # board_column_outputs = [feature[..., :20] for feature in raw_column_outputs]
        # active_column_outputs = [feature[..., 20:] for feature in raw_column_outputs]
        column_outputs = tf.keras.layers.concatenate(
            raw_column_outputs, axis=1
        )  # None x 10 x 20
        # active_outputs = tf.keras.layers.concatenate(
        #     active_column_outputs, axis=1
        # )  # None x 10 x 20
        tensor_dict["column_outputs"] = tf.transpose(
            column_outputs, [0, 2, 1]
        )  # None x 20 x 10
        # tensor_dict["active_outputs"] = tf.transpose(active_outputs, [0, 2, 1])

        # extract max height for more supervision during pretraining
        column_compressor = tf.keras.layers.Dense(8, activation="relu")
        row_compressor = tf.keras.layers.Dense(8, activation="relu")
        columns = tf.unstack(tensor_dict["column_outputs"], axis=2)  # 10 x [None x 40]
        column_specific_global_features = tf.keras.layers.concatenate(
            [row_compressor(col_feature) for col_feature in columns]
        )
        tensor_dict["column_global_features"] = column_compressor(
            column_specific_global_features
        )

        # track the active piece

        # tensor_dict["column_feat1"] = tf.keras.layers.Conv2D(
        #     16, (5, 1), padding="valid", activation="relu"
        # )(tensor_dict["transformed_input"])
        # tensor_dict["column_feat2"] = tf.keras.layers.Conv2D(
        #     16, (10, 1), padding="valid", activation="relu"
        # )(tensor_dict["column_feat1"])
        # tensor_dict["column_feat3"] = tf.keras.layers.Conv2D(
        #     8, (5, 1), padding="valid", activation="relu"
        # )(tensor_dict["column_feat2"])

        # construct output
        out_features = {
            feature_name: tensor_dict[feature_name] for feature_name in output_features
        }
        if concat:
            if len(out_features.values()) > 1:
                out_features = tf.keras.layers.concatenate(
                    [
                        tf.keras.layers.Flatten()(feature)
                        for feature in out_features.values()
                    ]
                )
            else:
                out_features = tf.keras.layers.Flatten()(list(out_features.values())[0])

            if downscale:
                out_features = tf.keras.layers.Dense(
                    TETRIS_EMBED_DIM, activation="relu"
                )(out_features)

        return tf.keras.Model(
            inputs=tensor_dict["input"], outputs=out_features, name="vision"
        )

    elif env_name in ("tetris", "tetris-simple"):
        inp = tf.keras.Input(shape=(20, 10, 3))
        ft = tf.keras.layers.Conv2D(4, (1, 1), activation=None)(inp)
        conv1 = tf.keras.layers.Conv2D(
            128, (5, 3), activation="relu", padding="same"  # 64
        )(ft)
        # ds1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding="same")(conv1)
        col = tf.keras.layers.Conv2D(16, (5, 1), padding="valid", activation="relu")(ft)
        col2 = tf.keras.layers.Conv2D(8, (5, 1), padding="valid", activation="relu")(
            col
        )
        col3 = tf.keras.layers.Conv2D(8, (11, 1), padding="valid", activation="relu")(
            col2
        )
        conv2 = tf.keras.layers.Conv2D(
            64, (9, 5), activation="relu", padding="same"  # 128, (5, 5)
        )(conv1)
        ds2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding="same")(conv2)
        conv3 = tf.keras.layers.Conv2D(
            64, (5, 3), activation="relu", padding="same"  # 128
        )(ds2)
        ds3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding="same")(conv3)
        conv4 = tf.keras.layers.Conv2D(32, (5, 3), activation="relu", padding="same")(
            ds3
        )
        # ds3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding="same")(conv3)
        small_features = tf.keras.layers.Flatten()(conv1)
        medium_features = tf.keras.layers.Flatten()(conv2)
        large_features = tf.keras.layers.Flatten()(conv3)
        global_features = tf.keras.layers.Flatten()(conv4)
        column_features = tf.keras.layers.Flatten()(col3)
        features = tf.keras.layers.concatenate(
            [
                small_features,
                medium_features,
                large_features,
                global_features,
                column_features,
            ]
        )
        fc0 = tf.keras.layers.Dense(2 * TETRIS_EMBED_DIM, activation="relu")(features)
        fc1 = tf.keras.layers.Dense(TETRIS_EMBED_DIM, activation="relu")(fc0)
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
            hidden = tf.keras.layers.Dense(512, activation="relu")(
                inp
            )  # get_normed_resblock(inp, TETRIS_EMBED_DIM, TETRIS_EMBED_DIM)
            hidden2 = tf.keras.layers.Dense(256, activation="relu")(
                hidden
            )  # get_normed_resblock(hidden, TETRIS_EMBED_DIM, TETRIS_EMBED_DIM)
            # hidden3 = get_normed_resblock(hidden2, TETRIS_EMBED_DIM, TETRIS_EMBED_DIM)
            hidden3 = tf.keras.layers.Dense(128, activation="relu")(hidden2)
            hidden4 = tf.keras.layers.Dense(64, activation="relu")(hidden3)
            val = tf.keras.layers.Dense(1, activation=None)(hidden4)
            adv = tf.keras.layers.Dense(14, activation=None)(hidden4)
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


def get_normed_resblock(inp, dim: int, inner_dim: int):
    hidden = tf.keras.layers.Dense(inner_dim, activation=None)(inp)
    # bn1 = tf.keras.layers.BatchNormalization()(hidden)
    out1 = tf.keras.layers.Activation("relu")(hidden)
    hidden2 = tf.keras.layers.Dense(dim, activation=None)(out1)
    # bn2 = tf.keras.layers.BatchNormalization()(hidden2)
    res = hidden2 + inp
    out = tf.keras.layers.Activation("relu")(res)
    return out


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
