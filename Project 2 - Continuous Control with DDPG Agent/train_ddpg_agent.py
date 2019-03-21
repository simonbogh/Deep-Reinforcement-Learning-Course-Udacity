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


'''Project 2 - Continuous Control with DDPG
In this project a double-jointed arm is trained to reach target locations. A
reward of +0.1 is provided for each step that the agent's hand is in the goal
location. Thus, the goal of the agent is to maintain its position at the target
location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position,
rotation, velocity, and angular velocities of the arm.

Each action is a vector with four numbers, corresponding to torque applicable
to two joints. Every entry in the action vector should be a number between
-1 and 1.

The Unity ML-Agents environment is employed for the training environment. For
this project, two separate versions of the *Reacher* Unity environment are
provided:
* Version 1: The first version contains a single agent
* Version 2: The second version contains 20 identical agents, each with its own
  copy of the environment

The task is episodic, and the environment is considered solved when the agent
gets an average score of +30 over 100 consecutive episodes.
'''

print('========================================')
# Create writer to write events to tensorboard using tensorboardX. Events are
# logged to the folder ./runs. In order to see the training process i.e. mean
# reward, launch tensorboard from the root folder by running:
#   $ tensorboard --logdir runs
writer = SummaryWriter()

# ### Importing the DDPG agent and Unity environment ### #
# Agent:
#  First the custom DDPG agent is imported from 'ddpg_agent.py'. 'ddpg_agent.py'
#  contains a class where a standard DQN agent has been designed. Details
#  regarding the agent design can be found in 'ddpg_agent.py'.
# Model:
#  The architecture of the model is described in 'ddpg_model.py'. Different
#  architectures for the neural network have been tried. The current version
#  employs an architecture with two fully-connected hidden layers and 96 units
#  in each layer for both the actor and critic.
from ddpg_agent import DDPGAgent

# Load Unity environment:
#  - Version 1: 'Reacher_1_agent'
#  - Version 2: 'Reacher_20_agents'
# Environments for different platforms (Windows, Linux, MacOS) can be found in
# the folder /unity-ml-agents/. There are also headless versions available for Linux
# env = UnityEnvironment(file_name='unity-ml-agents/Reacher_1_agent.app')
# env = UnityEnvironment(file_name='unity-ml-agents/Reacher_20_agents.app')

# Run without rendering: no_graphics=True
env = UnityEnvironment(file_name='unity-ml-agents/Reacher_20_agents.app', no_graphics=True)

# Environments contain **_brains_** which are responsible for deciding the
# actions of their associated agents. Here we check for the first brain
# available, and set it as the default brain we will be controlling from Python.
# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
print("Brain name:", brain_name)
print("Brain:", brain)

# Examine the State and Action Spaces
# In this environment, a double-jointed arm can move to target locations. A
# reward of `+0.1` is provided for each step that the agent's hand is in the
# goal location. Thus, the goal of your agent is to maintain its position at the
# target location for as many time steps as possible.
# The observation space consists of `33` variables corresponding to position,
# rotation, velocity, and angular velocities of the arm.  Each action is a
# vector with four numbers, corresponding to torque applicable to two joints.
# Every entry in the action vector must be a number between `-1` and `1`.

# Print out information about the state and action spaces
# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# Number of agents in the environment
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
# ### Training the DDPG agent to solve the environment ### #
# In the following the training takes place.
# When training the environment, setting `train_mode=True` accelerates the
# simulation environment so that training can be done much quicker.

# Make a new agent from the DDPGAgent class in 'ddpg_agent.py'
agent = DDPGAgent(state_size=state_size, action_size=action_size, num_agents=num_agents, random_seed=1) # , seed=random.randint(1,100000000)

# Print information about the network architecture
print('====================')
print('Actor-Critic network architecture')
print(agent.actor_local)
print(agent.critic_local)
print('====================')


# Set up a function that takes in information for training the agent for n_episodes.
def trainDDPG(n_episodes=1000, print_every=10):
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
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")

        # Save the actor and critic model parameters (weights and biases) to files
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

        # Print info to console
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        # Write mean score to tensorboard
        writer.add_scalar('data/scalar1', np.mean(score), i_episode)

    return scores

# Call function to start training for n_episodes
scores = trainDDPG(n_episodes=200)

print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

# Export tensorboard scalar data to JSON for external processing
writer.export_scalars_to_json("./all_scalars.json")
writer.close()

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