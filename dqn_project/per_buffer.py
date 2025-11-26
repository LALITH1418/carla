# per_buffer.py
import numpy as np

class PrioritizedReplayBuffer:
    """
    Proportional PER (numpy-based). Good for medium buffer sizes.
    alpha default set to 0.6 (standard prioritization).
    """
    def __init__(self, capacity=100000, alpha=0.6, epsilon=1e-5):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.epsilon = epsilon

        self.buffer = []
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity
            self.size += 1
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def _get_probabilities(self):
        prios = self.priorities[:self.size].astype(np.float64)
        probs = prios ** self.alpha
        total = probs.sum()
        if total == 0:
            probs = np.ones_like(probs) / probs.size
        else:
            probs = probs / total
        return probs

    def sample(self, batch_size, beta=0.4):
        if self.size == 0:
            return None

        probs = self._get_probabilities()
        replace = False if self.size >= batch_size else True
        indices = np.random.choice(self.size, batch_size, p=probs, replace=replace)

        N = self.size
        weights = (N * probs[indices]) ** (-beta)
        weights = weights / (weights.max() + 1e-8)

        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.uint8),
                indices,
                np.array(weights, dtype=np.float32))

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            prio = abs(float(td)) + self.epsilon
            if idx < self.capacity:
                self.priorities[idx] = prio

    def __len__(self):
        return self.size