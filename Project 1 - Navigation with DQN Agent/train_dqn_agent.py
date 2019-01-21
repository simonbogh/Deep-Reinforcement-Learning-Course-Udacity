from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch
import random
import pickle
import sys
import matplotlib as mpl
 # Set backend for image rendering, else the import will give an error in MacOS
mpl.use('TkAgg')
import matplotlib.pyplot as plt


'''Project 1 - Navigation with DQN Agent
In this project an agent is trained to collect yellow bananas (+1) while
avoiding blue bananas (-1) in a 3D world. The goal of the agent is to maximize
its reward by collecting as many yellow bananas as possible during one episode.

The Unity ML-Agents environment is employed for the training environment. The
simulation contains a single agent that navigates a large environment.

The state-space has 37 dimensions containing the agent's velocity and distance
measurements in front of the agent. Based on this information, the agent learns
how to apply four actions:
* 0: move forward.
* 1: move backward.
* 2: turn left.
* 3: turn right.

The reward function is based on providing +1 for a yellow banana, and -1 for a
blue banana.

The task is episodic, and the environment is considered solved when the agent
gets an average score of +13 over 100 consecutive episodes.

The solution employs a Deep Q-Network (DQN) and learns to collect an average
score of more than +13 after 280 episodes, but performance improves up to
around 900 episodes.
'''

print('========================================')
# ### Importing the DQN agent and Unity environment #### #
# Agent:
#  First the custom DQN agent is imported from 'agent.py'. 'agent.py' contains a
#  class where a standard DQN agent has been designed. Details regarding the
#  agent design can be found in 'agent.py'.
# Model:
#  The architecture of the model is described in 'model.py'. Different
#  architectures for the neural network has been tried. The current version
#  employs an architecture with two fully-connected hidden layers.
from agent import Agent

# Load Unity environment 'Banana'.
# Environments for different platforms (Windows, Linux, MacOS) can be found in
# the folder /unity-ml-agents/.
env = UnityEnvironment(file_name="unity-ml-agents/Banana.app")

# Unity environments contain **_brains_** which are responsible for deciding
# the actions of their associated agents. Here we check for the first brain
# available, and set it as the default brain we will be controlling.
# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
print("Brain name:", brain_name)
print("Brain:", brain)


# Print out information about the state and action spaces
# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# Number of agents in the environment
print('Number of agents:', len(env_info.agents))
# Number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)
# State space information
#  A two dimensional numpy array of dimension (batch size, vector observation size)
state = env_info.vector_observations[0]
print('State shape:', env_info.vector_observations.shape)   # (1, 37)
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)                    # 37


# ### Training the DQN agent to solve the environment ### #
# In the following the training takes place.
# When training the environment, setting `train_mode=True` accelerates the
# simulation environment so that training can be done much quicker.

# Make a new agent from the Agent class in 'agent.y'
agent = Agent(state_size=state_size, action_size=action_size, seed=random.randint(1,100000000))

# Print information about the network architecture
print('====================')
print('Network architecture')
print(agent.qnetwork_local)
print('====================')


# Set up a function that takes in information for training the agent for
# n_episodes. An ε-greedy policy is followed and that starting and ending
# value for ε can also be set. Finally, ε-decay is defined as an input
# argument.
def trainDQN(n_episodes=1000, eps_start=1.0, eps_end=0.1, eps_decay=0.995):
    """Deep Q-Learning agent that can be trained for a set amount of episodes

    Arguments:
        n_episodes {int} -- Number of episodes to train (default: {2000})
        eps_start {float} -- Start value for ε (default: {1.0})
        eps_end {float} -- End (min.) value for ε (default: {0.1})
        eps_decay {float} -- ε-decay value (default: {0.995})
    """

    # Initialise parameters
    scores = []                         # List with scores from each episode
    scores_window = deque(maxlen=100)   # Last 100 scores
    eps = eps_start                     # Initialize epsilon
    average_score = 0                   # Placeholder for the average score over a set number of episodes

    # Loop over n_episodes. For each episode the agent is trained until the
    # episode terminates. In this given scenario, the episode terminates after
    # a fixed t_max defined by the environment as 300. Thus the agent will train
    # until 'done' == True.
    print('Starting training ..')
    for i_episode in range(1, n_episodes+1):
        # Reset the environment when starting a new episode
        # 'train_mode=True': runs with small window and much faster
        env_info = env.reset(train_mode=True)[brain_name]   # Reset the environment.
        state = env_info.vector_observations[0]             # Get the current initial state
        score = 0                                           # Initialize the score
        step = 0                                            # Step in episode (used for printing)

        # Train until episode terminates (done == True)
        # Terminates automatically after 300 steps.
        while True:
            # Get action according to the current ε-greedy policy
            action = agent.act(state, eps)                  # Select ε-greedy action
            #action = np.random.randint(action_size)        # Select a random action

            # Step using action and get next state
            env_info = env.step(action)[brain_name]         # Send the action to the environment
            next_state = env_info.vector_observations[0]    # Get the next state

            # Get reward
            reward = env_info.rewards[0]                    # Get the reward
            done = env_info.local_done[0]                   # See if episode has finished

            # Step the agent learning and update the Q-network
            agent.step(state, action, reward, next_state, done)

            # Count score
            score += reward                                 # Update the score
            state = next_state                              # Roll over the state to next time step
            step += 1
            #print("\rStep {}".format(step), end="\r")
            #sys.stdout.flush()                             # Flush the terminal output

            # Exit loop if episode finished
            if done:
                break

        # Save scores (rewards) for later
        scores_window.append(score)         # Save most recent score in 100 last scores
        scores.append(score)                # Append score
        eps = max(eps_end, eps_decay*eps)   # Decay epsilon, account for epsilon minimum

        # Calculate the mean average score for the last 100 episodes
        new_average_score = np.mean(scores_window)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, new_average_score), end="")

        # For every 10th episode, check if the new average score is larger than
        # the old previous average score. If the new average score is larger
        # then a checkpoint of the current network weights are saved to a '.pth'
        # file. This ensures that we save the weights of the model where we
        # achieved the maximum average score for 100 consecutive episodes.
        if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon {:.2f}'.format(i_episode, new_average_score, eps))

            # Compare average scores
            if average_score < new_average_score:
                average_score = new_average_score
                # Save the network weights
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoints/checkpoint.pth')
                print('\r - Saving checkpoint with new average score: {:.2f}'.format(new_average_score))

    return scores


# Call function to start training for n_episodes
scores = trainDQN(n_episodes=1000)

# Close the unity environment when done
env.close()

# Finally the achived scores are saved to a file using the 'Pickle'
# framework. The scores can be used later on to e.g. plot graphs.
file_name = "scores.p"                    # Filename
file_object = open(file_name, 'wb')     # Open the with writing rights (w)
pickle.dump(scores, file_object)        # Save scores to file
file_object.close()                     # Close the file

# Plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

print('========================================')
print('Training done')
print('========================================')
# ### Evaluate the trained agent ### #
print(' - To evaluate the trained agent, check out \'evaluate_agent.py\'')
print('========================================')
