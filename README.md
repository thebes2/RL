## RL Playground

TODO:
- [x] Move training to EC2 and learn how to use remote jupyter (laptop is dying, do it even if there's no real perf benefit)
	- [x] Create virtualenv that works with jupyter notebook
	- [x] Create config files for current envs 
	
- [x] Analyze performance bottlenecks with lp

  - The neural network's inference is the bottleneck, it is almost 6x as slow as the environment step

  - Updating the network seems to be the bottleneck, taking up 75% of the time; Collecting rollouts takes the other 25%

    - How can we speed this up? The PPO loss fn is super slow because of bad implementation
    - [x] Speed up compute_policy_loss 
      - tf.map_fn is extremely slow bc it's serial, use tf.where to speed it up
      - tf.gather_nd is also slow for a similar reason, multiply by one hot vector and sum up to do parallelize (nvm. seems like feeding through the net is the bottleneck)

    - Now, the bottleneck is get_action in collect_rollout, need to use multithreading to speed it up further

- [x] Parallelize to utilize more cpu threads when collecting rollouts / gradients

  - [x] Look into the spinning up implementation (MPI)
  - Takes roughly 35s/epoch when single-threaded, 28s/epoch for two threads, 25s/epoch for four threads

- (Algo) The current agent gets ~10 apples per run on snake
  - [ ] It's decent, but it's not improving fast enough. The actions are too certain, might be overfitting and decreasing exploration. Add a noise function that decreases in amplitude over time to prevent this

- [ ] Analyze performance of tetris env
  - Runs at ~560 microseconds per step
  - Snake env runs at 460 microseconds, but with far less variation



Run MPI with 2 threads: `mpiexec -n 2 python -m rl.run_mpi`
