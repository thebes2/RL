import tensorflow as tf

def get_policy_architecture(env_name, algo='PPO'):
    if env_name == "CartPole-v0":
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
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        if algo != 'PPO':
            raise NotImplementedError("No architecture for {} yet".format(algo))
    elif env_name == "gym_snake:snake-v0":
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(15, 15, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax' if algo == 'PPO' else None)
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
        raise NotImplementedException()
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
    return value