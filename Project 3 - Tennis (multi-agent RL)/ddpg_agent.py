import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

# Import DDPG model from 'ddpg_model.py'
from ddpg_model import Actor, Critic

BUFFER_SIZE  = int(1e6) # replay buffer size
BATCH_SIZE   = 128      # minibatch size
GAMMA        = 0.99     # discount factor
TAU          = 1e-3     # for soft update of target parameters
LR_ACTOR     = 1.0e-4   # learning rate of the actor            standard: 1e-4
LR_CRITIC    = 1.0e-4   # learning rate of the critic           standard: 1e-3
WEIGHT_DECAY = 0        # L2 weight decay (regularisation??)

EPSILON       = 1.0     # explore->exploit noise process added to act step
EPSILON_DECAY = 0.0    # decay rate for noise process       standard: 1.5e-5
EPSILON_MIN   = 0.0     # minimum epsilon

UPDATE_EVERY = 1        # How many timesteps before doing an update. Tried with 20 timesteps, with no great success
NUM_UPDATES  = 1        # Number of update passes per timestep when updating from the experience replay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define DDPG agent
class DDPGAgent():
    """DDPG agent that interacts with and learns from the environment.

    The agents model is implemented in 'ddpg_model.py'. It consists of two
    neural networks; one for the actor, and one for the critic.

    The DDPGAgent class makes use of two other classes: ReplayBuffer, OUNoise
    """

    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.

        Arguments:
            state_size (int) -- dimension of each state
            action_size (int) -- dimension of each action
            num_agents (int) -- number of agents (brains)
            random_seed (int) -- random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        self.epsilon = EPSILON

        ### Make neural networks (local and target) for both actor and critic, and set optimizers
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise((num_agents, action_size), random_seed)

        # Initialize replay memory ###
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience in memory
        for i in range(self.num_agents):
            self.memory.add(state[i,:], action[i,:], reward[i], next_state[i,:], done[i])

        # Learn every UPDATE_EVERY time steps
        if timestep % UPDATE_EVERY == 0:
            # If we have collected enough experience in our memory i.e. more
            # than the mini-batch size, then call the self.learn() function
            if len(self.memory) > BATCH_SIZE:
                # Number of updates per timestep
                for _ in range(NUM_UPDATES):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy.

        Arguments:
            state {[type]} -- Current state
            add_noise {bool} -- Add noise (exploration) to the actions (default: {True})

        Returns:
            [float] -- Actions
        """

        # Convert 'state' numpy array to pytorch tensor using the current device
        # i.e. GPU or CPU.
        state = torch.from_numpy(state).float().to(device)

        # Set the module in evaluation mode.
        self.actor_local.eval()
        with torch.no_grad():
            # Evaluate the network with the current state
            action = self.actor_local(state).cpu().data.numpy()

        # Set the module in training mode.
        self.actor_local.train()
        if add_noise:
            # Add noise to the actions to add exploration
            action += self.epsilon * self.noise.sample()

        # Return the clipped actions
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Arguments:
            experiences {Tuple[torch.Tensor]} -- tuple of (s, a, r, s', done) tuples
            gamma {float} -- discount factor
        """

        # Experiences, mini-batch of 128
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- Update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Clip the gradients
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        # Take one step with the optimizer
        self.critic_optimizer.step()

        # ---------------------------- Update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- Update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        # ---------------------------- update noise ---------------------------- #
        self.epsilon -= EPSILON_DECAY
        if self.epsilon < EPSILON_MIN:
            self.epsilon = EPSILON_MIN
        # self.noise.reset()


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Arguments:
            local_model -- PyTorch model (weights will be copied from)
            target_model -- PyTorch model (weights will be copied to)
            tau (float) -- interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process.

        Arguments:
            size {[type]} -- [description]
            seed {[type]} -- [description]

        Keyword Arguments:
            mu {[type]} -- [description] (default: {0.})
            theta {float} -- [description] (default: {0.15})
            sigma {float} -- [description] (default: {0.2})
        """

        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

# Define replay memory class
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Arguments:
            buffer_size {int} -- Maximum size of buffer
            batch_size {int} -- Size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Sample a random mini-batch from the experience replay buffer
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        # Convert the sampled experiences to torch tensors on the GPU or CPU,
        # depending on what 'device' is. 'device' is set at the beggining of
        # the script.
        # The '.to()' method of Tensors and Modules can be used to easily move
        # objects to different devices (instead of having to call cpu() or
        # cuda() based on the context).
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        # Return tuple of random sampled experiences
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the length of the replay memory.

        Returns:
            [int] -- Length of the current replay memory
        """
        return len(self.memory)