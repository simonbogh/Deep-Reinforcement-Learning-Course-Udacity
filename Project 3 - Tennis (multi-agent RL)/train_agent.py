from unityagents import UnityEnvironment
from collections import deque
import numpy as np
import torch
import random
import pickle
import sys
import matplotlib as mpl
 # Set backend for image rendering, else the import will give an error in MacOS
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from itertools import count
from tensorboardX import SummaryWriter


'''Project 3 - Collaboration and Competition
In this project, two agents control rackets to bounce a ball over a net. If an
agent hits the ball over the net, it receives a reward of +0.1. If an agent lets
a ball hit the ground or hits the ball out of bounds, it receives a reward of
-0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and
velocity of the ball and racket. Each agent receives its own, local observation.

Two continuous actions are available, corresponding to movement toward (or away
from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get
an average score of +0.5 (over 100 consecutive episodes, after taking the
maximum over both agents). Specifically:
* After each episode, we add up the rewards that each agent received (without
discounting), to get a score for each agent. This yields 2 (potentially
different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of
those scores is at least +0.5.
'''

print('========================================')
# Create writer to write events to tensorboard using tensorboardX. Events are
# logged to the folder ./runs. In order to see the training process i.e. mean
# reward, launch tensorboard from the root folder by running:
#   $ tensorboard --logdir runs
writer = SummaryWriter()

# ### Importing the agent and Unity environment ### #
# Agent:
#  First the custom agent is imported from 'ddpg_agent.py'. 'ddpg_agent.py'
# Details regarding the agent design can be found in 'ddpg_agent.py'.
# Model:
#  The architecture of the model is described in 'ddpg_model.py'. Different
#  architectures for the neural network have been tried. The current version
#  employs an architecture with two fully-connected hidden layers and 256 units
#  in each layer for both the actor and critic.
from ddpg_agent import DDPGAgent


# If the code cell below returns an error, please revisit the project
# instructions to double-check that you have installed [Unity
# ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)
# and [NumPy](http://www.numpy.org/).

# Load Unity environment:
# Environments for different platforms (Windows, Linux, MacOS) can be found in
# the folder /unity-ml-agents/.

# Run without rendering: no_graphics=True
env = UnityEnvironment(file_name='unity-ml-agents/Tennis.app', no_graphics=False)

# Environments contain **_brains_** which are responsible for deciding the
# actions of their associated agents. Here we check for the first brain
# available, and set it as the default brain we will be controlling from Python.
# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
print("Brain name:", brain_name)
print("Brain:", brain)

# Examine the State and Action Spaces
# In this environment, two agents control rackets to bounce a ball over a net.
# If an agent hits the ball over the net, it receives a reward of +0.1.  If an
# agent lets a ball hit the ground or hits the ball out of bounds, it receives a
# reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.
# The observation space consists of 8 variables corresponding to the position
# and velocity of the ball and racket. Two continuous actions are available,
# corresponding to movement toward (or away from) the net, and jumping.

# Print out information about the state and action spaces
# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# Number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)
# Size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)
# State space information
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


print('========================================')
# FOR TESTING RANDOM ACTIONS
# Take Random Actions in the Environment
# In the next code cell, you will learn how to use the Python API to control the
# agents and receive feedback from the environment.
# Once this cell is executed, you will watch the agents' performance, if they
# select actions at random with each time step.  A window should pop up that
# allows you to observe the agents.
# Of course, as part of the project, you'll have to change the code so that the
# agents are able to use their experiences to gradually choose better actions
# when interacting with the environment!

# for i in range(1, 6):                                      # play game for 5 episodes
#     env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
#     states = env_info.vector_observations                  # get the current state (for each agent)
#     scores = np.zeros(num_agents)                          # initialize the score (for each agent)
#     while True:
#         actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
#         actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
#         env_info = env.step(actions)[brain_name]           # send all actions to tne environment
#         next_states = env_info.vector_observations         # get next state (for each agent)
#         rewards = env_info.rewards                         # get reward (for each agent)
#         dones = env_info.local_done                        # see if episode finished
#         scores += env_info.rewards                         # update the score (for each agent)
#         states = next_states                               # roll over states to next time step
#         if np.any(dones):                                  # exit loop if episode finished
#             break
#     print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))


print('========================================')
# Make a new agent from the DDPGAgent class in 'ddpg_agent.py'
agent = DDPGAgent(state_size=state_size, action_size=action_size, num_agents=num_agents, random_seed=1) # , seed=random.randint(1,100000000)

# Print information about the network architecture
print('====================')
print('Network architecture')
print(agent.actor_local)
print(agent.critic_local)
print('====================')



# Set up a function that takes in information for training the agent for n_episodes.
def trainAgents(n_episodes=1000, print_every=10):
    '''Deep Deterministic Policy Gradient (DDPG) agent that can be trained for a set amount of episodes.

    Keyword Arguments:
        n_episodes {int} -- Number of episodes to train (default: {1000})
        print_every {int} -- How often to print information to the console (default: {10})

    Returns:
        [[float]] -- Episode scores
    '''

    # Initialise parameters
    scores = []                                 # List with scores from each episode and agent
    scores_deque = deque(maxlen=print_every)    # Last XXX scores

    # Loop over n_episodes. For each episode the agent is trained until the episode terminates.
    for i_episode in range(1, n_episodes+1):
        # Reset the environment when starting a new episode
        # 'train_mode=True': runs with small window and much faster
        env_info = env.reset(train_mode=True)[brain_name]   # Reset the environment
        state = env_info.vector_observations                # Get the current initial state for all agents
        agent.reset()                                       # Reset agent noise
        score = np.zeros(num_agents)                        # Initialize the score

        # Train until episode terminates (done == True)
        for t in count():
            action = agent.act(state, add_noise=True)

            # Step using action and get next state
            env_info = env.step(action)[brain_name]         # Send the action to the environment
            # next_state = env_info.vector_observations[0]  # Get the next state
            next_state = env_info.vector_observations       # Get the next state

            # Get reward
            reward = env_info.rewards                       # Get the reward
            done = env_info.local_done                      # See if episode has finished

            # next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done, t)
            state = next_state                              # Roll over the state to next time step
            score += reward                                 # Update the score

            # Exit loop if episode finished
            if np.any(done):
                break

        # Save scores (rewards) for later
        scores_deque.append(np.max(score))
        scores.append(np.max(score))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")

        # Save the actor and critic model parameters (weights and biases) to files
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

        # Print info to console
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        # Write max score to tensorboard
        writer.add_scalar('data/scalar1', np.max(score), i_episode)
        # writer.add_scalar('data/epsilon', agent.epsilon, i_episode)

    return scores

# Call function to start training for n_episodes
scores = trainAgents(n_episodes=10000)

print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


# Close the unity environment when done
env.close()

# Plot the scores
# plt.plot(np.arange(1, len(scores)+1), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()

print('========================================')
print('Training done')
print('========================================')