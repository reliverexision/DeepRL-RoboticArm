# DDPG+HER & TD3+HER to pick Strawberries

## 1. Introduction
This project aims to implement and compare the performance of Deep Deterministic Policy Gradient (DDPG) and Twin Delayed DDPG (TD3) with Hindsight Experience Replay (HER) Extension using an edited  MuJoCo's robotic FetchPickAndPlace environment to pick strawberries.

## 2. Installation Guide
This project requires the use of MuJoCo. Please follow the installation instructions found [here](https://github.com/openai/mujoco-py) to install.
```shell
pip3 install -r requirements.txt
```

## 3. Usage Guide

### 3.1 To Train the Strawberry Picker
1. Navigate to the file below and open it with a text editor of your choice.
```shell
nano ME5046/main.py
```
2. Change the following parameters as shown below.
```shell
Train         = True
Play_FLAG     = False 
```
3. Run the following command.
```shell
mpirun -np $(nproc) python3 -u main.py
```

### 3.2. To Watch the Trained Strawberry Picker
1. Navigate to the file below and open it with a text editor of your choice.
```shell
nano ME5046/main.py
```
2. Change the following parameters as shown below.
```shell
Train         = False
Play_FLAG     = True 
```
3. Run the following command.
```shell
python3 main.py
```

## 4. Demo of Trained Strawberry Picker
The GIF shows the trained strawberry picker putting the strawberries into the basket.
<p align="center">
  <img src="Demo/FetchPickAndPlace.gif" height=250>
</p>  


## 5. Result
### 5.1 DDPG+HER
<p align="center">
  <img src="Result/DDPG_Fetch_PickandPlace.jpg" height=400>
</p>

### 5.2 TD3+HER
<p align="center">
  <img src="Result/TD3_Fetch_PickandPlace.png" height=400>
</p>

## 6. References
1. [_Continuous control with deep reinforcement learning_, Lillicrap et al., 2015](https://arxiv.org/abs/1509.02971)  
2. [_Hindsight Experience Replay_, Andrychowicz et al., 2017](https://arxiv.org/abs/1707.01495)
3. [_Addressing Function Approximation Error in Actor-Critic Methods_, Fujimoto et al., 2018](https://arxiv.org/pdf/1802.09477.pdf)
4. [_Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research_, Plappert et al., 2018](https://arxiv.org/abs/1802.09464)
5. [_Deep Deterministic Policy Gradient (DDPG): Theory and Implementation_, Guha, 2020](https://towardsdatascience.com/deep-deterministic-policy-gradient-ddpg-theory-and-implementation-747a3010e82f)
6. [_TD3: Learning To Run With AI_, Byrne, 2019](https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93)

## 7. Acknowledgements
Credit goes to [@alirezakazemipour](https://github.com/alirezakazemipour) for [his original implementation in PyTorch](https://github.com/alirezakazemipour/DDPG-HER) of [@TianhongDai's](https://github.com/TianhongDai) [simplified implementation of HER](https://github.com/TianhongDai/hindsight-experience-replay) of [the original OpenAI's code](https://github.com/openai/baselines/tree/master/baselines/her).
