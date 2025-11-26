# dqn_agent.py - STABILIZED VERSION (Standard DQN)
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from per_buffer import PrioritizedReplayBuffer
import os

class DQNNet(nn.Module):
    """Deeper network with layer normalization for stability."""
    def __init__(self, state_size, action_size):
        super(DQNNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.LayerNorm(256),  # Add normalization
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self,
                 state_size,
                 action_size,
                 device=None,
                 lr=1e-4,                # Increased from 3e-5
                 gamma=0.99,             # Standard discount
                 batch_size=64,          # Reduced for more frequent updates
                 buffer_size=100000,     # Sufficient size
                 epsilon_start=1.0,
                 epsilon_min=0.05,       # Lower minimum for more exploitation
                 epsilon_decay=0.995,    # Balanced decay
                 alpha=0.6,              # Standard PER alpha
                 beta_start=0.4,         # Standard beta start
                 beta_frames=100000,     # Complete annealing
                 update_every=4,         # Update every N steps (not every step)
                 target_update_freq=1000, # Soft update frequency
                 tau=0.001,              # Soft update rate
                 model_dir="models"):

        self.state_size = state_size
        self.action_size = action_size
        self.device = device if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.buffer = PrioritizedReplayBuffer(capacity=buffer_size, alpha=alpha)

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame_idx = 0

        # Networks
        self.model = DQNNet(state_size, action_size).to(self.device)
        self.target_model = DQNNet(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # Always in eval mode

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)
        
        # Huber loss for robustness
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

        # Update frequencies
        self.update_every = update_every
        self.target_update_freq = target_update_freq
        self.learn_steps = 0
        self.step_count = 0
        
        self.tau = tau
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Track metrics
        self.losses = []

    def act(self, state, exploit=False):
        """Select action using epsilon-greedy policy."""
        if (not exploit) and (random.random() < self.epsilon):
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_t)
            action = q_values.argmax(dim=1).item()
        return action

    def remember(self, s, a, r, ns, d):
        """Store experience in replay buffer."""
        self.buffer.add(s, a, r, ns, d)
        self.step_count += 1

    def _update_beta(self):
        """Linearly anneal beta from beta_start to 1.0."""
        self.frame_idx += 1
        frac = min(1.0, self.frame_idx / float(self.beta_frames))
        self.beta = self.beta_start + frac * (1.0 - self.beta_start)

    def replay(self):
        """Train on a batch from replay buffer using Standard DQN."""
        # Only update every N steps
        if self.step_count % self.update_every != 0:
            return
        
        if len(self.buffer) < self.batch_size:
            return

        self._update_beta()
        batch = self.buffer.sample(self.batch_size, beta=self.beta)
        if batch is None:
            return

        states, actions, rewards, next_states, dones, indices, is_weights = batch

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        is_weights_t = torch.FloatTensor(is_weights).to(self.device)

        # Current Q values
        current_q = self.model(states_t).gather(1, actions_t).squeeze(1)

        # Target Q values (Standard DQN)
        with torch.no_grad():
            # Use target network to get max Q value for next states
            next_q = self.target_model(next_states_t).max(1)[0]
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        # Compute TD errors for PER
        td_errors = (target_q - current_q).detach().cpu().numpy()

        # Compute weighted loss
        loss_per_sample = self.loss_fn(current_q, target_q)
        weighted_loss = (is_weights_t * loss_per_sample).mean()

        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update priorities in buffer
        self.buffer.update_priorities(indices, td_errors)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Soft update target network
        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self._soft_update_target()
        
        self.losses.append(weighted_loss.item())

    def _soft_update_target(self):
        """Soft update target network parameters."""
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, name="dqn_ckpt.pth"):
        """Save model checkpoint."""
        path = os.path.join(self.model_dir, name)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_steps': self.learn_steps,
            'step_count': self.step_count
        }, path)
        return path

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', {}))
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.learn_steps = checkpoint.get('learn_steps', 0)
        self.step_count = checkpoint.get('step_count', 0)
        print(f"Loaded checkpoint: epsilon={self.epsilon:.3f}, steps={self.step_count}")