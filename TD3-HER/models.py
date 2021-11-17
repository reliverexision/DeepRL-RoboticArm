from torch import nn
import torch
from torch.nn import functional as F
import numpy as np


def init_weights_biases(size):
    v = 1.0 / np.sqrt(size[0])
    return torch.FloatTensor(size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, n_goals, n_hidden1=256, n_hidden2=256, n_hidden3=256, initial_w=3e-3):
        self.n_states = n_states[0]
        self.n_actions = n_actions
        self.n_goals = n_goals
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.initial_w = initial_w
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(in_features=self.n_states + self.n_goals, out_features=self.n_hidden1)
        self.fc2 = nn.Linear(in_features=self.n_hidden1, out_features=self.n_hidden2)
        self.fc3 = nn.Linear(in_features=self.n_hidden2, out_features=self.n_hidden3)
        self.output = nn.Linear(in_features=self.n_hidden3, out_features=self.n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = torch.tanh(self.output(x))  # TODO add scale of the action

        return output


class Critic(nn.Module):
    def __init__(self, n_states, n_goals, n_hidden1=256, n_hidden2=256, n_hidden3=256, initial_w=3e-3, action_size=1):
        super(Critic, self).__init__()

        self.n_states = n_states[0]
        self.n_goals = n_goals

        self.initial_w = initial_w
        self.action_size = action_size

        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3

        self.n_hidden4 = n_hidden1
        self.n_hidden5 = n_hidden2
        self.n_hidden6 = n_hidden3
        
        # Q1 Architecture
        self.fc1 = nn.Linear(in_features=self.n_states + self.n_goals + self.action_size, out_features=self.n_hidden1)
        self.fc2 = nn.Linear(in_features=self.n_hidden1, out_features=self.n_hidden2)
        self.fc3 = nn.Linear(in_features=self.n_hidden2, out_features=self.n_hidden3)

        # Q2 Architecture
        self.fc4 = nn.Linear(in_features=self.n_states + self.n_goals + self.action_size, out_features=self.n_hidden4)
        self.fc5 = nn.Linear(in_features=self.n_hidden4, out_features=self.n_hidden5)
        self.fc6 = nn.Linear(in_features=self.n_hidden5, out_features=self.n_hidden6)

        self.output1 = nn.Linear(in_features=n_hidden3, out_features=1)
        self.output2 = nn.Linear(in_features=n_hidden3, out_features=1)

    def forward(self, x, a):
        x1 = F.relu(self.fc1(torch.cat([x, a], dim=-1)))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))

        x2 = F.relu(self.fc4(torch.cat([x, a], dim=-1)))
        x2 = F.relu(self.fc5(x2))
        x2 = F.relu(self.fc6(x2))
        output1 = self.output1(x1)
        output2 = self.output2(x2)

        return output1, output2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.fc1(xu))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)
        return x1
