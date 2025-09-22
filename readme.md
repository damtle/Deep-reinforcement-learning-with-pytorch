**Status:** Active (under active development, breaking changes may occur)

This repository will implement the classic and state-of-the-art deep reinforcement learning algorithms. The aim of this repository is to provide clear pytorch code for people to learn the deep reinforcement learning algorithm. 

In the future, more state-of-the-art algorithms will be added and the existing codes will also be maintained.

![demo](https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/figures/grid.gif)

## Requirements
- python <=3.6 
- tensorboardX
- gymnasium >= 0.29
- pytorch >= 2.0

## Unified training framework (Python 3.10 + Gymnasium)

The repository now ships with a modular training framework that unifies the algorithms included in the book chapters under a
single configuration-driven entry point.  Common utilities such as replay buffers, rollout storage and training loops live in
`rl_framework/` and can be shared across value-based and policy-based agents.  Switching the algorithm or the environment is as
simple as selecting another YAML configuration file.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Running an experiment

```bash
# Deep Q-Network on CartPole-v1
python train.py --config configs/dqn_cartpole.yaml --total-episodes 10

# REINFORCE / Policy Gradient on CartPole-v1
python train.py --config configs/pg_cartpole.yaml --total-episodes 10

# Advantage Actor-Critic variants
python train.py --config configs/actor_critic_cartpole.yaml --total-episodes 10
python train.py --config configs/a2c_cartpole.yaml --total-episodes 10

# Off-policy continuous-control algorithms
python train.py --config configs/ddpg_pendulum.yaml --total-episodes 5
python train.py --config configs/sac_pendulum.yaml --total-episodes 5
python train.py --config configs/td3_pendulum.yaml --total-episodes 5

# ACER on CartPole-v1
python train.py --config configs/acer_cartpole.yaml --total-episodes 10

# Proximal Policy Optimisation on CartPole-v1
python train.py --config configs/ppo_cartpole.yaml --total-episodes 10
```

Each configuration file contains three sections:

- `environment`: the Gymnasium environment id, optional arguments and episode horizon.
- `algorithm`: the agent name (`dqn`, `ppo`, …) together with its hyper-parameters.
- `training`: generic trainer options such as number of episodes, evaluation cadence, logging frequency and target device.

Algorithms dynamically adapt to the environment spaces.  Replay-based agents such as DQN, DDPG, SAC and TD3 share the same
high-performance buffer implementation, while policy gradient families (Policy Gradient, Actor-Critic, A2C, ACER, PPO) reuse the
rollout utilities and actor-critic networks.  The framework is fully compatible with Python 3.10 and the latest Gymnasium API
(returning `(observation, info)` from `reset` and `(observation, reward, terminated, truncated, info)` from `step`).

Supported algorithms and example configuration files:

- `dqn` – `configs/dqn_cartpole.yaml`
- `policy_gradient` – `configs/pg_cartpole.yaml`
- `actor_critic` – `configs/actor_critic_cartpole.yaml`
- `a2c` – `configs/a2c_cartpole.yaml`
- `ddpg` – `configs/ddpg_pendulum.yaml`
- `ppo` – `configs/ppo_cartpole.yaml`
- `acer` – `configs/acer_cartpole.yaml`
- `sac` – `configs/sac_pendulum.yaml`
- `td3` – `configs/td3_pendulum.yaml`

## DQN

Here I uploaded two DQN models which is trianing CartPole-v0 and MountainCar-v0.

### Tips for MountainCar-v0

This is a sparse binary reward task. Only when car reach the top of the mountain there is a none-zero reward. In genearal it may take 1e5 steps in stochastic policy. You can add a reward term, for example, to change to the current position of the Car is positively related. Of course, there is a more advanced approach that is inverse reinforcement learning.

![value_loss](https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char01%20DQN/DQN/pic/value_loss.jpg)   
![step](https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char01%20DQN/DQN/pic/finish_episode.jpg) 
This is value loss for DQN, We can see that the loss increaded to 1e13, however, the network work well. Because the target_net and act_net are very different with the training process going on. The calculated loss cumulate large. The previous loss was small because the reward was very sparse, resulting in a small update of the two networks.

### Papers Related to the DQN


  1. Playing Atari with Deep Reinforcement Learning [[arxiv]](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/1.dqn.ipynb)
  2. Deep Reinforcement Learning with Double Q-learning [[arxiv]](https://arxiv.org/abs/1509.06461) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/2.double%20dqn.ipynb)
  3. Dueling Network Architectures for Deep Reinforcement Learning [[arxiv]](https://arxiv.org/abs/1511.06581) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/3.dueling%20dqn.ipynb)
  4. Prioritized Experience Replay [[arxiv]](https://arxiv.org/abs/1511.05952) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/4.prioritized%20dqn.ipynb)
  5. Noisy Networks for Exploration [[arxiv]](https://arxiv.org/abs/1706.10295) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb)
  6. A Distributional Perspective on Reinforcement Learning [[arxiv]](https://arxiv.org/pdf/1707.06887.pdf) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/6.categorical%20dqn.ipynb)
  7. Rainbow: Combining Improvements in Deep Reinforcement Learning [[arxiv]](https://arxiv.org/abs/1710.02298) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/7.rainbow%20dqn.ipynb)
  8. Distributional Reinforcement Learning with Quantile Regression [[arxiv]](https://arxiv.org/pdf/1710.10044.pdf) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/8.quantile%20regression%20dqn.ipynb)
  9. Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation  [[arxiv]](https://arxiv.org/abs/1604.06057) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/9.hierarchical%20dqn.ipynb)
  10. Neural Episodic Control [[arxiv]](https://arxiv.org/pdf/1703.01988.pdf) [[code]](#)


## Policy Gradient


Use the following command to run a saved model


```
python Run_Model.py
```


Use the following command to train model


```
python pytorch_MountainCar-v0.py
```



> policyNet.pkl

This is a model that I have trained.


## Actor-Critic

This is an algorithmic framework, and the classic REINFORCE method is stored under Actor-Critic.
 
## DDPG  
Episode reward in Pendulum-v0:  

![ep_r](https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char05%20DDPG/DDPG_exp.jpg)  


## PPO  

- Original paper: https://arxiv.org/abs/1707.06347
- Openai Baselines blog post: https://blog.openai.com/openai-baselines-ppo/


## A2C

Advantage Policy Gradient, an paper in 2017 pointed out that the difference in performance between A2C and A3C is not obvious.

The Asynchronous Advantage Actor Critic method (A3C) has been very influential since the paper was published. The algorithm combines a few key ideas:

- An updating scheme that operates on fixed-length segments of experience (say, 20 timesteps) and uses these segments to compute estimators of the returns and advantage function.
- Architectures that share layers between the policy and value function.
- Asynchronous updates.

## A3C

Original paper: https://arxiv.org/abs/1602.01783

## SAC

**This is not the implementation of the author of paper!!!**

Episode reward in Pendulum-v0:

![ep_r](https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char09%20SAC/SAC_ep_r_curve.png)

## TD3

**This is not the implementation of the author of paper!!!**  

Episode reward in Pendulum-v0:  

![ep_r](https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char10%20TD3/TD3_Pendulum-v0.png)  

Episode reward in BipedalWalker-v2:  
![ep_r](https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char10%20TD3/Episode_reward_TD3_BipedakWalker.png)  

If you want to use the test your model:

```
python TD3_BipedalWalker-v2.py --mode test
```

## Papers Related to the Deep Reinforcement Learning
[01] [A Brief Survey of Deep Reinforcement Learning](https://arxiv.org/abs/1708.05866)  
[02] [The Beta Policy for Continuous Control Reinforcement Learning](https://www.ri.cmu.edu/wp-content/uploads/2017/06/thesis-Chou.pdf)  
[03] [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  
[04] [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)  
[05] [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)  
[06] [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)  
[07] [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/abs/1603.00748)  
[08] [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)  
[09] [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)  
[10] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)  
[11] [Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation](https://arxiv.org/abs/1708.05144)  
[12] [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)  
[13] [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)  
[14] [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)  

## TO DO
- [x] DDPG
- [x] SAC
- [x] TD3


# Best RL courses
- [OpenAI's spinning up](https://spinningup.openai.com/)  
- [David Silver's course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)  
- [Berkeley deep RL](http://rll.berkeley.edu/deeprlcourse/)  
- [Practical RL](https://github.com/yandexdataschool/Practical_RL)  
- [Deep Reinforcement Learning by Hung-yi Lee](https://www.youtube.com/playlist?list=PLJV_el3uVTsODxQFgzMzPLa16h6B8kWM_)   
