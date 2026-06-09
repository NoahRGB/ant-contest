import torch

import agents
import env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_MEMORY_NODES = 6
TRAJECTORY_LENGTH = 30
EPISODES = 20000

agents = [
    agents.BPTTAgent(use_sensor=True, num_memory_nodes=NUM_MEMORY_NODES, device=device),
    agents.BPTTAgent(use_sensor=True, num_memory_nodes=NUM_MEMORY_NODES, device=device),
]

env.train(EPISODES, TRAJECTORY_LENGTH, NUM_MEMORY_NODES, agents, device, None)