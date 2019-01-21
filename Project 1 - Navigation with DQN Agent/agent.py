# %%
import numpy as np
import random

# namedtuple:   factory function for creating tuple subclasses with named fields
# deque:        list-like container with fast appends and pops on either end
from collections import namedtuple, deque

# Custom DQN model set up in model.py
from model import QNetwork

# Import torch framework
import torch
import torch.nn.functional as F
import torch.optim as optim

# %%
# Set hyperparameters
BUFFER_SIZE = int(100000)   # Replay memory size
BATCH_SIZE = 64             # Mini-batch size
GAMMA = 0.99                # Discount factor
TAU = 0.001                 # For soft update of target parameters
LR = 0.0001                 # Learning rate alpha
UPDATE_EVERY = 4            # How often to update the network

# Check if we have CUDA enabled device i.e. a GPU, else use the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define DQN agent
class Agent():
    '''Agent to interact with the environment and learn
    '''

    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        ### Make Q-network, both the local and the target network ###
        self.qnetwork_local = QNetwork(state_size, action_size, seed)
        self.qnetwork_target = QNetwork(state_size, action_size, seed)

        ### Set optimizer for gradient descent ###
        # Select the parameters of the local qnetwork we are training on.
        # Weights will later be copied to the target qnetwork
        self.optimizer = optim.Adam(params=self.qnetwork_local.parameters(), lr=LR)

        ### Initialize replay memory ###
        self.replay_memory = ReplayMemory(buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)

        ### Initialize time step (for updating every UPDATE_EVERY steps)###
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        ### Save experience in memory ###
        self.replay_memory.add(state, action, reward, next_state, done)

        ### Learn every UPDATE_EVERY time steps ###
        self.t_step += 1
        # If remainder of division is 0, then learn from experience replay memory
        if self.t_step % UPDATE_EVERY == 0:
            # If we have collected enough experience in our memory i.e. more
            # than the mini-batch size, then call the self.learn() function
            if len(self.replay_memory) > BATCH_SIZE:
                experiences = self.replay_memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.0):
        """Return actions for the given state and policy (e.g. ε-greedy)

        Arguments:
            state {[numpy array]} -- Current state
            eps {float} -- Epsilon for ε-greedy policy (default: {0.0})
        """
        # Convert 'state' numpy array to pytorch tensor using the current device
        # i.e. GPU or CPU.
        # unsqueeze() returns a new tensor with a dimension of size one
        # inserted at the specified position.
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # Set the module in evaluation mode.
        self.qnetwork_local.eval()
        with torch.no_grad():
            # Evaluate the Q-network with the current state
            # Call a forward pass with the current state to get q-values for
            # that state.
            action_values = self.qnetwork_local(state)

        # Set the module in training mode.
        self.qnetwork_local.train()

        # ε-greedy action selection
        if random.random() > eps:
            # Get action with the max q-value
            # .cpu(): Move all model parameters and buffers to the CPU
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples

        Arguments:
            experiences {Tupple(torch.Tensor)} -- Tuple of (s, a, r, s', done) tuples
            gamma {float} -- Discount factor
        """
        # Experiences, mini-batch of 64
        states, actions, rewards, next_states, dones = experiences

        ### Get max predicted Q-values (for next states) from target model ###
        # .detach(): returns a new Variable that does not back-propagate to
        # whatever detach() was called on. Creates a tensor that shares storage
        # with tensor that does not require grad.
        # .max(): return the max value of a tensor on a given dimension (dim)
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # size([4, 1])
        # the tensors below are [64, x]
        ### Computer Q-targets for current states ###
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        ### Get expected Q-values from local model ###
        # .gather(): Gathers values along an axis specified by dim.
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        ### Compute loss ###
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize loss
        # Clear the gradients so they are not accumulated
        self.optimizer.zero_grad()
        loss.backward()
        # Perform a single optimization step.
        self.optimizer.step()


        ### Update target network ###
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters

        Arguments:
            local_model {PyTorch model} -- Copy weights from
            target_model {PyTorch model} -- Copy weights to
            tau {float} -- Interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau) * target_param.data)


# Define replay memory class
class ReplayMemory:
    def __init__(self, buffer_size, batch_size):
        """Initialise a replay memory buffer
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add new experience to the replay memory buffer

        Arguments:
            state {[type]} -- Current state
            action {[type]} -- Current action
            reward {[type]} -- Reward for taking action in current state
            next_state {[type]} -- Next state after taking action
            done {bool} -- Whether episode is done (terminal state)
        """
        new_experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(new_experience)

    def sample(self):
        """Sample a random mini-batch from the experience replay buffer
        """
        sampled_experiences = random.sample(self.memory, k=self.batch_size)

        # Convert the sampled experiences to torch tensors on the GPU or CPU,
        # depending on what 'device' is. 'device' is set at the beggining of
        # the script.
        # The '.to()' method of Tensors and Modules can be used to easily move
        # objects to different devices (instead of having to call cpu() or
        # cuda() based on the context).
        states = torch.from_numpy(np.vstack([e.state for e in sampled_experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in sampled_experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in sampled_experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in sampled_experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in sampled_experiences if e is not None]).astype(np.uint8)).float().to(device)

        # Return tuple of random sampled experiences
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the length of the replay memory

        Returns:
            [int] -- Length of the current replay memory
        """
        return len(self.memory)
