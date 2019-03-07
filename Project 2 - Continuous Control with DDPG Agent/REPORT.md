# Project 2 - Report
## Continuous Control with DDPG agent

### Implementation Details
<!--Description of the implementation.-->
For the second project the actor-critic method **Deep Deterministic Policy Gradient (DDPG)** was applied.

Paper: [Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).](https://arxiv.org/abs/1509.02971)

### Learning Algorithm
<!-- Clearly describe the learning algorithm, along with the chosen hyperparameters. -->
<!--Also describe the model architectures for any neural networks.-->
**Deep Deterministic Policy Gradient (DDPG)**
DDPG consists of value-based and policy-based side and therefore implements two neural networks. DDPG is an off-policy algorithm, and can be used for environments with continous action spaces.

The model architecture consists of an actor and critic network. The neural networks consists of an input layer of size 33, two fully-connected hidden layers, and an output layer of size 4 for the actor and size 1 for the critic.

Initially I tried to solve Version 1 (1 agent) first, but training time was long and it took a long time to tune hyper-parameters. Therefore I switched to Version 2 (20 agents) in the hope to speed up training and then later switch back to Version 1. This was a success as can be seen in the training graphs below in the Results section.

Different sizes were tried for the two hidden layers: (48,48), (96,48), (96,96), (128,128), and (256,256). The different layer sizes were evaluated until the training started to show some convergence. (96,96) performed the best and was hereafter fixed. 

ReLU (Rectified Linear Unit) is used for the activation function in the actor's hidden layers and tanh on the output layer. ReLU was also applied on the hidden layers in the critic's hidden layers, and no activation on the output.

Exploration - Adding noise to the actions: To make DDPG explore better, noise is added to the actions at training time. Ornstein–Uhlenbeck (OU) noise is used in this implementation as suggested in the original paper. At evaluation time, noise is not added to the actions.

In order to stabilise the learning, gradient clipping was applied to see if it would have any effect on the performance and convergence time. It did however, as can be seen in the training graphs below, not a have significant effect.

Soft updates of target parameters we introduced as well as experience replay, as these two tricks are what was shown in DQN to stabilise learning when using neural networks as function approximators. 

Finally, when convergence was achieved I tuned the learning rate as one the final things, and this had a huge positiv impact on convergence time. Initially a learning rate of 1e-4 was used for the actor, and 1e-3 for the critic. In the next training session both learning rates were set to 1e-4, which made a huge improvement as can be seen in the graphs below. Then I lowered the learning even more to 0.8e-4 and finally 0.6e-4. This stabilised the training a lot and it converged after 56 steps where it achieved a mean score of 30+ for the 100 consecutive episodes

After the good results with Version 2 and 20 agents, the same model architecture and hyper-parameters were used for Version 1 with one agent. The results are a bit mixed here. As can be seen in the result section, it managed to solve the environment within 300 episodes, but other training runs also diverged and never achieved good results. The graphs below are smoothed, and the original training graph can be seen in the background. It is seen that training Version 1 is really unstabled and the reward per episode oscillates. 

**Significant changes that affected performance**
* Learning rate
* Number of nodes in hidden layers in neural networks

## Results
<!-- Video: [YouTube.com](https://youtu.be/laOg6DYBc6c) -->
<!--Plot of Rewards-->

Below are the training results from Version 1 and 2 of the environment. First, Version 2 was solved. Afterwards Version 1 was trained with an identical network architecture and hyper-parameters as Version 2.

A plot of rewards per episode is included below to illustrate that:

* [version 1] the agent receives an average reward (over 100 episodes) of at least +30
* [version 2] the agent is able to receive an average reward (over 100 episodes, and over all 20 agents) of at least +30

Version 1 was solved after 300 episodes. Version 2 was solved after 56 episodes.

### Version 2: 20 agents
![training_results](images/training_results_20_agents.jpg)

<!-- <img src="images/training_results.jpg" alt="Training Results" width="300"> -->

### Version 1: 1 agent
![training_results](images/training_results_1_agent_vs_20_agents.jpg)


## Ideas for Future Work
<!--For improving the agent in the future, several ideas can be implemented e.g **Double DQN** to cope with overestimation, **Dueling DQN** to decouple the value and advantage, and also **Prioritized Experience Replay** can be interesting in order to make better use of the stored experience.-->
<!---->
<!--Distributed Training-->
<!---->
<!--The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.-->
