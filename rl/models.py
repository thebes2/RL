import tensorflow as tf

def get_policy_architecture(env_name, algo='PPO'):
    if env_name == "CartPole-v0":
        if 'Dueling' in algo:
            inp = tf.keras.Input(shape=(4,))
            hidden1 = tf.keras.layers.Dense(16, activation='relu')(inp)
            hidden2 = tf.keras.layers.Dense(16, activation='relu')(hidden1)
            val = tf.keras.layers.Dense(1, activation='linear')(hidden2)
            adv = tf.keras.layers.Dense(2, activation='linear')(hidden2)
            avg = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, 1))(adv)
            out = tf.keras.layers.Add()([val, adv, -avg])
            model = tf.keras.Model(inputs=inp, outputs=out, name="dueling-DQN")
        else:
            model = tf.keras.Sequential([
                tf.keras.Input(shape=(4,)),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax' if algo == 'PPO' else None)
            ])
    elif env_name == "MountainCar-v0":
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(2,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax' if algo == 'PPO' else None)
        ])
    elif env_name == "Acrobot-v1":
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(6,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(48, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax' if algo == 'PPO' else None)
        ])
    elif env_name == "gym_snake:snake-v0":
        if 'Dueling' in algo:
            inp = tf.keras.Input(shape=(15, 15, 3,))
            ft = tf.keras.layers.Conv2D(3, (1,1), activation=None)(inp)
            conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(ft)
            conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv1)
            features = tf.keras.layers.Flatten()(conv2)
            hidden1 = tf.keras.layers.Dense(512, activation='relu')(features)
            hidden2 = tf.keras.layers.Dense(64, activation='relu')(hidden1)
            val = tf.keras.layers.Dense(1, activation='linear')(hidden2)
            adv = tf.keras.layers.Dense(4, activation='linear')(hidden2)
            avg = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, 1))(adv)
            out = tf.keras.layers.Add()([val, adv, -avg])
            model = tf.keras.Model(inputs=inp, outputs=out, name="dueling-DQN")
        else:
            model = tf.keras.Sequential([
                tf.keras.Input(shape=(15, 15, 3)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(4, activation='softmax' if 'PPO' in algo else None)
            ])
    elif env_name == "tetris":  # the final raid boss
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(20,10,3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='elu'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='elu', padding='valid'), # new addition
            tf.keras.layers.Conv2D(128, (3, 3), activation='elu'),
            tf.keras.layers.Flatten(),
            #tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(256, activation='elu'),
            tf.keras.layers.Dense(64, activation='elu'),
            tf.keras.layers.Dense(7, activation='softmax') # NO-OP is an action
        ])
        if algo != 'PPO':
            raise NotImplementedError("No architecture for {} yet".format(algo))
    elif env_name == "Breakout-v0":
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(210, 160, 3,)),
            tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu'),
            tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu'),
            tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(4, activation=None)
        ])
        """
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(42,32,3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='elu'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='elu', padding='valid'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='elu'),
            tf.keras.layers.Dense(64, activation='elu'),
            tf.keras.layers.Dense(3, activation='softmax' if algo == 'PPO' else None)
        ])
        """
    elif env_name == "LunarLander-v2":
        if 'Dueling' in algo:
            inp = tf.keras.Input(shape=(8,))
            hidden1 = tf.keras.layers.Dense(128, activation='relu')(inp)
            hidden2 = tf.keras.layers.Dense(128, activation='relu')(hidden1)
            hidden3 = tf.keras.layers.Dense(64, activation='relu')(hidden2)
            val = tf.keras.layers.Dense(1, activation='linear')(hidden3)
            adv = tf.keras.layers.Dense(4, activation='linear')(hidden3)
            avg = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, 1))(adv)
            out = tf.keras.layers.Add()([val, adv, -avg])
            model = tf.keras.Model(inputs=inp, outputs=out, name="dueling-DQN")
        else:
            model = tf.keras.Sequential([
                tf.keras.Input(shape=(8,)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(4, activation='softmax' if 'PPO' in algo else None)
            ])
    return model

def get_value_architecture(env_name):
    if env_name == "CartPole-v0":
        value = tf.keras.Sequential([
            tf.keras.Input(shape=(4,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation=None)
        ])
    elif env_name == "MountainCar-v0":
        raise NotImplementedError()
    elif env_name == "Acrobot-v1":
        value = tf.keras.Sequential([
            tf.keras.Input(shape=(6,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(48, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation=None)
        ])
    elif env_name == "gym_snake:snake-v0":
        value = tf.keras.Sequential([
            tf.keras.Input(shape=(15, 15, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            #tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            #tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation=None)
        ])
    elif env_name == "tetris":  # the final raid boss
        value = tf.keras.Sequential([
            tf.keras.Input(shape=(20,10,3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='elu'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='elu', padding='valid'), # new addition
            tf.keras.layers.Conv2D(128, (3, 3), activation='elu'),
            tf.keras.layers.Flatten(),
            #tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(256, activation='elu'),
            tf.keras.layers.Dense(64, activation='elu'),
            tf.keras.layers.Dense(1, activation=None)
        ])
    elif env_name == "LunarLander-v2":
        value = tf.keras.Sequential([
            tf.keras.Input(shape=(8,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation=None)
        ])
    return value