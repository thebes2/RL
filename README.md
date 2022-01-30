## RL Playground

A collection of RL algorithms.

### Running a model

For running a general model:
```
python -m rl.run --env=<env>
```
where `<env>` is the name of a configuration file under `configs/` (like `snake_spr`).

For running a model using explicit multithreading:
```
mpiexec -n <n_threads> python -m rl.run_mpi
```
which splits up rollouts and gradient computation onto `<n_threads>` threads. Currently, this only makes sense for `PPO` as the training bottleneck for `DQN`-like agents are the update steps (which TensorFlow parallelizes across threads) and not environment interaction.

For running a model on GPU: Coming soon:tm:

### Setup
1. Clone the repo
2. In the root of the repo, run `scripts/setup.sh`
3. Ensure you have `pre-commit` installed. Run `pre-commit install` in the root of the repo
