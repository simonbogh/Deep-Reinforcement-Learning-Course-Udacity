# Project 1 - Navigation with a DQN agent
This folder contains the solution to project 1 on navigation for the Udacity Nanodegree.

In this project an agent is trained to **collect yellow bananas (+1)** while avoiding blue bananas (-1) in a 3D world. The goal of the agent is to maximize its reward by collecting as many yellow bananas as possible during one episode. The **Unity ML-Agents** environment is employed for the training environment. The simulation contains a single agent that navigates a large environment. The solution implements a **Deep Q-Network (DQN)** and learns to collect an average score of more than +13 after 280 episodes, but performance improves up to around 900 episodes.

## Project Details

### Environment Details
The state space has 37 dimensions containing the agent's velocity and distance measurements in front of the agent. Based on this information, the agent learns how to apply four actions:
* 0: move forward.
* 1: move backward.
* 2: turn left.
* 3: turn right.

The task is episodic, and the environment is considered solved when the agent gets an average score of +13 over 100 consecutive episodes.

### Learning Algorithm
Clearly describe the learning algorithm, along with the chosen hyperparameters.

Also describe the model architectures for any neural networks.

The neural network implemented consists of one input layer, two fully-connected hidden layers, and one output layer.

The reward function is based on providing +1 for a yellow banana, and -1 for a blue banana respectively.


## Results
Video: [YouTube.com](https://youtu.be/laOg6DYBc6c)

**Untrained agent:**

Below you see the untrained agent taking random actions in the environment. The agent does not know how to act in the environment, or that bananas yields a reward (+1 or -1). Next, you see the final trained agent, and how it performs during final training and evaluation.

![untrained](images/untrained_agent.gif)

**Trained agent:**

Below you see the agent after training. After several episodes of interacting with the environment, the agent learns that it yields good reward to collect the yellow bananas.

The submission reports the number of episodes needed to solve the environment.

![trained](images/trained_agent.gif)

![scores_untrained](images/scores_during_training.jpg)

Show scores with pure evaluation.

## Getting Started
### Installation
Instructions for installing dependencies or downloading needed files.

### Running the code
Describe how to run the code in the repository, to train and evaluate the agent.

**Training the agent**

    $ python train_agent.py

**Evaluating the agent**

    $ python evaluate_agent.py

## Ideas for Future Work
Concrete future ideas for improving the agent's performance.

## License
GPL-3.0

## Author
Simon Bøgh
