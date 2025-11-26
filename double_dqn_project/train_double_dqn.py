from carla_env import CarlaEnv
from double_dqn_agent import DoubleDQNAgent
import numpy as np

EPISODES = 2000
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

env = CarlaEnv()
state = env.reset()

agent = DoubleDQNAgent(
    state_size=len(state),
    action_size=env.action_space
)

epsilon = EPSILON_START

for episode in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state, epsilon)
        next_state, reward, done = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        state = next_state
        total_reward += reward

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    print(f"Episode {episode} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")
