import torch
import torch.nn as nn
import torch.nn.functional as F

'''Model class containing the network for the model being trained by the agent
'''


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialise parameters and set up model

        Arguments:
            state_size {[type]} -- Number of inputs
            action_size {[type]} -- Number of actions
            seed {int} -- Seed

        Keyword Arguments:
            fc1_units {int} -- Number of units in first hidden layer (default: {64})
            fc2_units {int} -- Number of units in second hidden layer (default: {64})
        """
        super(QNetwork, self).__init__()
        #self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Forward pass to get logits. 'Logits' is the raw output without e.g. softmas
        or similar. The logits are used to calcuclate the loss in the loss/cost function

        Arguments:
            state {tensor} -- Current state experienced by the agent

        Returns:
            [tensor???] -- Returns logits from the forward pass
        """

        # ReLU
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Leaky ReLU
        # x = F.leaky_relu(self.fc1(state), negative_slope=0.2)
        # x = F.leaky_relu(self.fc2(x), negative_slope=0.2)

        logits = self.fc3(x)             # No e.g. softmax, return logits, raw output
        return logits
