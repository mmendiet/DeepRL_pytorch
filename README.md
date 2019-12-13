# Transfer Learning Between Different Games for Prediction via Generative Methods
This repository is a modified version of [DeepRL_pytorch](https://github.com/wuyx/DeepRL_pytorch) for use in my Reinforcement Learning course project. The aim of this project is to quantify transfer learning and cross-game compatibility in Atari.

Utilized algorithms:
* Deep Q-Learning (DQN)
* Action Conditional Video Prediction


# Dependency
> Tested in macOS 10.12 and CentO/S 6.8
* Open AI gym
* [Roboschool](https://github.com/openai/roboschool) (Optional)
* PyTorch v0.3.0
* Python 2.7 
* [TensorboardX](https://github.com/lanpa/tensorboard-pytorch)


# Usage
```dataset.py```: generate dataset for action conditional video prediction

```main.py```: all other algorithms

# References
* [Human Level Control through Deep Reinforcement Learning](https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
* [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
* [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
* [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
* [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
* [HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](https://arxiv.org/abs/1106.5730)
* [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)
* [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
* [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
* [Hybrid Reward Architecture for Reinforcement Learning](https://arxiv.org/abs/1706.04208)
* [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
* [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
* [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286)
* [Action-Conditional Video Prediction using Deep Networks in Atari Games](https://arxiv.org/abs/1507.08750)
* [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
* [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)
