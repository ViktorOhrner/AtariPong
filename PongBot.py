import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import cv2
import time
import json
import gc

from DuelCNN import DuelCNN
import ale_py
gym.register_envs(ale_py)


class PongBot:
    def __init__(self, metadata):
        self.load_model = metadata['load_model']
        self.load_model_episode = metadata['load_model_episode']
        self.save_model = metadata['save_model']
        self.model_path = metadata['model_path']
        self.save_interval = metadata['save_interval']
        self.episodes = metadata['episodes']
        self.max_steps = metadata['max_steps']
        self.memory_length = metadata['memory_length']
        self.batch_size = metadata['batch_size']
        self.learning_rate = metadata['learning_rate']
        self.discount_factor_g = metadata['discount_factor_g']
        self.tau = metadata['tau']
        self.epsilon_decay_rate = metadata['epsilon_decay_rate']
        self.min_epsilon = metadata['min_epsilon']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CUDA error fixes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear CUDA cache
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False  # Disable benchmark for consistency
            torch.backends.cudnn.deterministic = True  # More stable but slower

        self.policy_dqn = DuelCNN().to(self.device)
        self.optimizer = optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate)

    def train(self):
        env = gym.make('PongDeterministic-v4')
        memory = deque([], maxlen=self.memory_length)
        target_dqn = DuelCNN().to(self.device)
        target_dqn.load_state_dict(self.policy_dqn.state_dict())
        target_dqn.eval()
        epsilon = 0.9
        epsilon_history = np.zeros(self.episodes)
        losses = []

        # Training loop
        for episode in range(0 + self.load_model_episode, self.episodes):
            start_time = time.time()
            state = env.reset()[0]
            state = self.pre_process(state)
            state = np.stack((state, state, state, state))
            terminated = None
            truncated = None
            steps = 0
            score = {'Opponent': 0, "PongBot": 0}

            for step in range(self.max_steps):
                if epsilon > random.random():
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = self.act(state)

                # Take step in environment
                new_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                new_state = self.pre_process(new_state)
                new_state = np.stack((new_state, state[0], state[1], state[2]))
                memory.append((state, action, new_state, reward, done))
                state = new_state
                steps += 1
                if reward == 0:
                    pass
                elif reward == 1:
                    score["PongBot"] += 1
                else:
                    score["Opponent"] += 1


                # Train - optimize the network
                if len(memory) >= self.batch_size:
                    # clear gradients before training
                    self.optimizer.zero_grad()

                    states, actions, next_states, rewards, dones = zip(*random.sample(memory, self.batch_size))

                    # Convert to tensors
                    states = torch.stack([torch.from_numpy(arr) for arr in states], dim=0).to(self.device, non_blocking=True)
                    actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)  # Shape: (batch_size, 1)
                    next_states = torch.stack([torch.from_numpy(arr) for arr in next_states], dim=0).to(self.device, non_blocking=True)
                    rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)  # Shape: (batch_size,)
                    dones = torch.tensor(dones, dtype=torch.float, device=self.device)  # Shape: (batch_size,)

                    # Ensure CUDA synchronization
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    # Predictions
                    state_q_values = self.policy_dqn(states)
                    with torch.no_grad():
                        next_states_q_values = self.policy_dqn(next_states)
                        next_states_target_q_values = target_dqn(next_states)

                    # Find selected Q-values
                    selected_q_values = state_q_values.gather(1, actions).squeeze(1)
                    # Select best actions in state st+1 using policy network
                    best_next_actions = next_states_q_values.max(1)[1].unsqueeze(1)
                    # Use greedy foe policy so it's called off-policy
                    next_states_target_q_value = next_states_target_q_values.gather(1, best_next_actions).squeeze(1) # Changed
                    # Use Bellman function to find expected q value
                    expected_q_values = rewards + self.discount_factor_g * next_states_target_q_value * (1 - dones)

                    # calculate loss
                    loss = F.mse_loss(selected_q_values, expected_q_values.detach())
                    #loss = (selected_q_values - expected_q_values.detach()).pow(2).mean()

                    # Backpropagation step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    losses.append(loss)

                    # Update target network
                    target_dqn_state_dict = target_dqn.state_dict()
                    policy_dqn_state_dict = self.policy_dqn.state_dict()
                    for key in policy_dqn_state_dict:
                        target_dqn_state_dict[key] = target_dqn_state_dict[key] * (1 - self.tau) + policy_dqn_state_dict[key] * self.tau
                    target_dqn.load_state_dict(target_dqn_state_dict)

                # End the episode if it is over, this is VERY important!    
                if done:
                    break

            # Memory cleanup every 10 episode
            if episode % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            # Update epsilon
            epsilon = max(epsilon * self.epsilon_decay_rate, self.min_epsilon)
            epsilon_history[episode] = epsilon

            # Print info
            if episode%1==0:
                current_time = time.time()
                print(f"{episode}: loss: {loss} epsilon: {epsilon}, steps: {steps}, latest ation: {action}, score: {score}, time:{current_time-start_time}")

            # Save the model every on interval
            if episode%self.save_interval == 0:
                policy_dqn_state_dict = self.policy_dqn.state_dict()
                save_path = self.model_path + str(episode) + ".pth"
                torch.save(policy_dqn_state_dict, save_path)
                #self.save_model(episode)

    def pre_process(self, image):
        # Grey scale
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Crop frame
        frame = frame[25: -10, :]
        # Rescale
        frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        # Normalize
        frame = frame / 255
        #return frame
        return torch.tensor(frame, dtype=torch.float32)
    
    def act(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.policy_dqn(state).argmax().item()
    
    def save_model_path(self, episode):
        policy_dqn_state_dict = self.policy_dqn.state_dict()
        save_path = self.model_path + str(episode) + ".pth"
        torch.save(policy_dqn_state_dict, save_path)
    
    def load_state_dict(self, state_dict_path):
        self.policy_dqn.load_state_dict(torch.load(state_dict_path))

    def get_state_dict(self):
        return self.policy_dqn.state_dict()
    
if __name__ == "__main__":
    metadata = {
        'load_model' : True,
        'load_model_episode' : 0,
        'model_path' : "./models/pong-cnn-duel-",
        'train_model' : True,
        'episodes' : 100000,
        'max_steps' : 100000,
        'memory_length' : 50000,
        'batch_size': 64,
        'gamma' : 0.97,
        'learning_rate' : 0.00025,
        'discount_factor_g' : 0.97,
        'tau' : 0.005,
        'epsilon_decay_rate' : 0.99,
        'min_epsilon' : 0.05,
        'save_model' : True,
        'save_interval' : 100,
        'render' : False
    }


    pong_bot = PongBot(metadata)

    if metadata['load_model']:
        model_load_path = metadata['model_path'] + str(metadata['load_model_episode']) + '.pth'
        print(model_load_path)
        #print(model_load_path)
        pong_bot.load_state_dict(model_load_path)

    if metadata['train_model']:
        pong_bot.train()

    # If the last interval did not save the model we do it now
    if metadata['save_model'] and metadata['episodes'] % metadata['save_interval'] == 0:
        print(metadata['episodes'])
        pong_bot.save_model_path(metadata['episodes'])
