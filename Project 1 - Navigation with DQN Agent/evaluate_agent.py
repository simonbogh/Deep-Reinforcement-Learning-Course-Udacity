from unityagents import UnityEnvironment
import torch
import random
from agent import Agent


print('========================================')
'''Evaluate agent: load trained weights from checkpoint file
'''
# Path to trained model weights
saved_model_weights = 'model_64x64/checkpoint_mean_14_76_at_780_random_seed.pth'

# Load environment
env = UnityEnvironment(file_name="unity-ml-agents/Banana.app")

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# Number of actions
action_size = brain.vector_action_space_size

# Get state size
state = env_info.vector_observations[0]
state_size = len(state)

# Load the agent
agent = Agent(state_size=state_size, action_size=action_size, seed=random.randint(1,100000000)) #
print('====================')
print('Network architecture')
print(agent.qnetwork_local)
print('====================')

# Load the saved weights and let it run in the environment
print('Loading checkpoint ..')
agent.qnetwork_local.load_state_dict(torch.load(saved_model_weights))

# Start evaluation in simulator
print('Starting evaluation ..')

for i in range(10):
    score = 0
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    while True:
        action = agent.act(state, eps=0.02)             # Could add a little epsilon, just in case so it does not get stuck
        env_info = env.step(action)[brain_name]         # send the action to the environment
        next_state = env_info.vector_observations[0]    # get the next state
        reward = env_info.rewards[0]                    # get the reward
        done = env_info.local_done[0]                   # episode done if True
        state = next_state
        score += reward
        print('\rScore: {:.0f}'.format(score), end="")
        if done:
            print('\rFinal score: {:.0f}'.format(score))
            break

# Close the unity environment when done
env.close()
