import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, x):
        return self.layers(x)


class DoubleDQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=1e-4,
                 buffer_size=50000, batch_size=64, tau=0.001):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.memory = deque(maxlen=buffer_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.update_target_network(tau=1.0)  # Hard copy at start

    def update_target_network(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, policy_param in zip(self.target_net.parameters(),
                                              self.policy_net.parameters()):
            target_param.data.copy_(
                tau * policy_param.data + (1 - tau) * target_param.data
            )

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).to(self.device)
        q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor([x[0] for x in minibatch]).to(self.device)
        actions = torch.LongTensor([x[1] for x in minibatch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([x[2] for x in minibatch]).to(self.device)
        next_states = torch.FloatTensor([x[3] for x in minibatch]).to(self.device)
        dones = torch.FloatTensor([x[4] for x in minibatch]).to(self.device)

        # ----------- DOUBLE DQN UPDATE -----------
        # action from policy network
        next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)

        # target Q-value from target network
        next_q = self.target_net(next_states).gather(1, next_actions).squeeze()

        target_q = rewards + (1 - dones) * self.gamma * next_q

        # current Q
        current_q = self.policy_net(states).gather(1, actions).squeeze()

        loss = nn.MSELoss()(current_q, target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target_network()
