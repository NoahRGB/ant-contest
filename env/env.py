import torch
import torch.optim as optim
import numpy as np

def food_density(pos, food_locations):
    # pos (num_ants, 2)
    # food_locations (num_ants, 2)
    # food_heights (num_ants,)

    bump_width = 8

    food_positions = food_locations[:, 0:2]
    food_heights = food_locations[:, 2]

    food_densities = food_heights.unsqueeze(1) + torch.exp(-torch.sum((pos - food_positions)**2, dim=1, keepdim=True) / (bump_width))

    return food_densities.squeeze() # (num_ants,)

def run_physics(s, a, food_locations, rand_food, device):

    if rand_food:
        # rand food
        # s (num_ants, 2 + 1 + num_memory_nodes)
        # a ((num_ants, 2), (num_ants, num_memory_nodes,), (num_ants, num_memory_nodes,))

        action, h_inp, h_gate = a
        initial_pos = s[:, 0:2] # (num_ants, 2)
        last_hidden = s[:, 3:] # (num_ants, num_memoy_nodes,)

        magnitudes = action.norm(dim=1, keepdim=True)
        directions = action / (magnitudes + 1e-6)
        velocity = 0.2 * torch.tanh(magnitudes) * directions
        next_pos = initial_pos + velocity

        next_hidden = torch.tanh(h_inp) * torch.sigmoid(h_gate) + last_hidden * (1 - torch.sigmoid(h_gate))

        foods = food_density(next_pos, food_locations) # (num_ants,)
        next_sensor = foods.unsqueeze(1) # (num_ants, 1)

        sprime = torch.cat([
            next_pos,
            next_sensor,
            next_hidden,
        ], dim=1).to(device) # (num_ants, 2 + 1 + num_memory_nodes)

        return foods, sprime

    else:
        # fixed food
        # s (num_ants, 2)
        # a (num_ants, 2)
        action = a[0]
        magnitudes = action.norm(dim=1, keepdim=True)
        directions = action / (magnitudes + 1e-6)
        velocity = 0.2 * torch.tanh(magnitudes) * directions
        sprime = s + velocity

        rewards = food_density(sprime, food_locations) # (num_ants,)

        return rewards, sprime

def expand_full_trajectory(trajectory_length, num_ants, rand_food, agent, device, initial_state, food_locations):
    
    total_rewards = torch.zeros(num_ants, dtype=torch.float32, requires_grad=True).to(device)
    s = initial_state

    for t in range(0, trajectory_length):
        action = agent.act(s) # (num_ants, action_dim)
        rewards, sprime = run_physics(s, action, food_locations, rand_food, device) # (num_ants,), (num_ants, state_dim)
        agent.update(s, sprime, action, rewards, torch.full((num_ants,), False, dtype=torch.float32).to(device))
        s = sprime
        total_rewards += rewards

    avg_total_reward = torch.mean(total_rewards)
    return avg_total_reward

def train(episodes, trajectory_length, num_ants, rand_food, num_memory_nodes, agent, device, writer, quiet=False):

    reward_history = []

    if rand_food: # if random food, keep start point at (0, 0)
        initial_pos = torch.zeros((num_ants, 2), dtype=torch.float32).to(device) # (num_ants, 2)

    for episode in range(episodes):
        
        if rand_food: 

            # if rand food, randomise food locations
            food_locations = torch.cat([
                torch.tensor(np.random.uniform(-4, 4, size=(num_ants, 2)), dtype=torch.float32).to(device),
                torch.tensor(np.random.uniform(0, 0, size=(num_ants, 1)), dtype=torch.float32).to(device),
            ], dim=1).to(device) # (num_ants, 3)

            # if rand food, add sensor + dummy memory
            initial_state = torch.cat([
                initial_pos,
                food_density(initial_pos, food_locations).unsqueeze(1),
                torch.zeros((num_ants, num_memory_nodes), dtype=torch.float32).to(device)
            ], dim=1).to(device)

        else: 

            # if fixed food, keep food at (0, 0)
            food_locations = torch.cat([
                torch.zeros((num_ants, 2), dtype=torch.float32).to(device),
                torch.tensor(np.random.uniform(0, 0, size=(num_ants, 1)), dtype=torch.float32).to(device),
            ], dim=1).to(device) # (num_ants, 3)

            # if fixed food, randomise start point
            initial_pos = torch.tensor(np.random.uniform(-4, 4, size=(num_ants, 2)), dtype=torch.float32).to(device) # (num_ants, 2)

            initial_state = initial_pos


        avg_reward = expand_full_trajectory(trajectory_length, num_ants, 
                                            rand_food, agent, device, initial_state, 
                                            food_locations)
        agent.finish_episode(avg_reward)

        cpu_reward = avg_reward.detach().cpu().numpy()
        reward_history.append(cpu_reward)
        if writer is not None: writer.add_scalar("avg_reward", cpu_reward, episode)
        if not quiet: print(f"episode {episode+1}/{episodes} avg_reward: {cpu_reward}")

    return reward_history
