import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import agents

class AntBrain(nn.Module):

    def __init__(self, num_inputs, num_outputs, num_memory_nodes):
        super(AntBrain, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_memory_nodes = num_memory_nodes

        self.fc1 = nn.Linear(num_inputs, 20)
        self.fc2 = nn.Linear(20, num_outputs)

    def forward(self, input):
        output = F.tanh(self.fc1(input))
        output = self.fc2(output)
        if self.num_memory_nodes > 0:
            return output[:, :2], output[:, 2:2+self.num_memory_nodes], output[:, 2+self.num_memory_nodes:]
        else:
            return (output,)

class BPTTAgent(agents.Agent):

    def __init__(self, use_sensor, num_memory_nodes, device):
        self.use_sensor = use_sensor
        self.num_memory_nodes = num_memory_nodes
        self.device = device

        self.network_input = 2 + (1 if use_sensor else 0) + num_memory_nodes
        self.network_output = 2 + 2 * num_memory_nodes

        self.action_network = AntBrain(self.network_input, self.network_output, num_memory_nodes).to(self.device)
        self.optimiser = optim.Adam(self.action_network.parameters(), lr=0.001)

    def act(self, s):
        return self.action_network(s)

    def update(self, s, sprime, a, r, done):
        pass

    def finish_episode(self, avg_reward):
                
        self.optimiser.zero_grad()
        loss = -avg_reward
        loss.backward()
        self.optimiser.step()