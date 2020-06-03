# Deep-Reinforcement-Learning-Hands-On-Second-Edition
Deep-Reinforcement-Learning-Hands-On-Second-Edition, published by Packt

## Quick-start

1. Install [PyTorch](https://pytorch.org/)

2. Install [ptan](https://github.com/Shmuma/ptan) from source

3. Clone this repository

4. ```cd /Deep-Reinforcement-Learning-Hands-On-Second-Edition/Chapter19```

5. ```python3 04_train_ppo.py -n pendulum```

## Changes made by simondlevy

So far I've only changed the code in Chapter19 (Trust Regions): 

* Removed RoboSchool dependency

* Made [Pendulum-v0](https://gym.openai.com/envs/Pendulum-v0/) the default environment

* Added command-line options for number of hidden units, maximum episodes, maximum run-time
