## Configs

Configuration files for running different tasks.

### Specification

Environment:
- `env_name`: name of the environment to be run on
- `env`: string for instantiating the environment from openai gym api
- `use_raw_env`: use raw environment rather than wrapper (allows overriding episode lengths)
- `n_actions`: number of output actions for the environment (currently only supports discrete action spaces)
- `t_max`: maximum length of episode

Training:
- `learning_rate`: learning rate for main models
- `minibatch_size`: number of rollouts to perform before updates (on-policy algorithms only)
- `batch_size`: number of samples to draw per update step (off-policy algorithms only)
- `multistep`: use multistep transitions for biased but faster training
- `update_freq`: number of environment interaction/frames per update step
- `update_step`: number of gradient steps (PPO-like algorithms only)

General:
- `train_epochs`: number of epochs to train (for DQN, this is the number of episodes the model sees, for PPO, the model will see `train_epochs * minibatch_size` episodes)
- `max_buf_size`: maximum size of the replay buffer (for off-policy only)
- `buf_size`: maximum number of transitions before a training cycle (for on-policy only)
- `algo`: list of parameters for indicating the model that will be trained (choices: `PPO`, `DQN`, `DDQN`, `Dueling`, `PER`, `SPR`)
- `callbacks`: list of callbacks for modifying model behavior

Hyperparameters:
- Exploration:
  - `epsilon`: probability of taking random action when exploring with epsilon-greedy strategy
- Rewards:
  - `gamma`: per-timestep discount factor
- Prioritized replay/sampling:
  - `alpha`: exponential coefficient for sampling probabilities (increases bias but mines harder samples)
  - `beta`: importance sampling coefficient (decreases bias but lowers learning rate)
- Target network updates:
  - `delta`: coefficient for target network update as average of online network parameters (set to `1.0` to use hard updates)
  - `target_delay`: number of updates before target network updates (for hard updates only)
- Self-predictive representations:
  - `latent_dim`: dimension of the latent representation used in transition model
  - `_multistep`: number of steps to unroll transition model
  - `lambda`: weighting coefficient for SPR loss when training

### Configs

The default configuration is stored in `default.yaml`. Defining a parameter in a task-specific configuration file will override the default value.

The default config creates a double DQN agent with dueling, Polyak updates for target network, and initializes the buffer with random episodes.
