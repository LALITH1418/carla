# train_dqn.py - STABILIZED VERSION
from dqn_agent import DQNAgent
from carla_env import CarlaEnv
import time
import numpy as np
import csv
import os
from collections import deque
import pygame

def init_keybaord():
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("CARLA manual control")

def get_keyboard_action():
    action = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        action = 0  # Turn left
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        action = 2  # Turn right
    elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
        action = 3  # Brake
    elif keys[pygame.K_w] or keys[pygame.K_UP]:
        action = 1  # Go straight
    return action

def collect_human_data(env, agent, episodes = 5, max_steps = 10000):
    """Collect episodes of human driving data for pretraining."""
    init_keybaord()
    print(f"Collecting {episodes} episodes of human driving data...")
    print("Use arrow keys or WASD to control the vehicle.")
    print("close small window to stop data collection.")

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        print(f"\n[Human Episode {ep}]")

        while not done and steps < max_steps:
            action = None
            # Wait until some valid action is pressed
            while action is None:
                action = get_keyboard_action()
                pygame.time.wait(10)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

        print(f"  Collected {steps} steps, total reward {total_reward:.2f}")

    pygame.quit()
    print("Human data collection finished.")
    print(f"Buffer size is now: {len(agent.buffer)}")

def run_eval_episode(env, agent, max_steps=500):
    """Run one greedy episode (no exploration) for evaluation."""
    state = env.reset()
    total_reward = 0.0
    steps = 0
    done = False

    # Temporarily force greedy behavior
    while not done and steps < max_steps:
        action = agent.act(state, exploit=True)  # exploit=True => no epsilon
        next_state, reward, done = env.step(action)

        state = next_state
        total_reward += reward
        steps += 1

    return total_reward, steps

def train():
    # Initialize environment
    env = CarlaEnv()
    state_size = None
    action_size = env.action_size

    # Initialize agent with stabilized hyperparameters
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=1e-4,
        gamma=0.99,
        batch_size=16,
        buffer_size=100000,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=100000,
        update_every=4,          # Key: don't train every step
        target_update_freq=1000,
        tau=0.001,
        model_dir="models"
    )
    collect_human_data(env, agent, episodes=5, max_steps=1000)
    # Training parameters
    EPISODES = 1000
    SAVE_EVERY = 50
    LOG_CSV = "training_log.csv"
    WARMUP_STEPS = 5000  # Collect experiences before training
    
    # Metrics tracking
    reward_window = deque(maxlen=100)
    step_window = deque(maxlen=100)

    # Initialize CSV log
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "total_reward", "avg_reward_100", 
                           "epsilon", "beta", "steps", "avg_loss"])

    print("=" * 60)
    print("STARTING STABILIZED DQN TRAINING")
    print("=" * 60)
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Device: {agent.device}")
    print(f"Warmup steps: {WARMUP_STEPS}")
    print("=" * 60)

    try:
        global_step = len(agent.buffer)
        
        for episode in range(1, EPISODES + 1):
            state = env.reset()
            total_reward = 0.0
            done = False
            steps = 0
            episode_start = time.time()

            while not done and steps < 1000:  # Increased max steps
                # Select action
                action = agent.act(state)
                
                # Take step
                next_state, reward, done = env.step(action)
                
                # Store transition
                agent.remember(state, action, reward, next_state, done)
                
                # Train agent (only after warmup and every N steps)
                if global_step > WARMUP_STEPS:
                    agent.replay()
                
                state = next_state
                total_reward += reward
                steps += 1
                global_step += 1

            episode_time = time.time() - episode_start
            
            # Track metrics
            reward_window.append(total_reward)
            step_window.append(steps)
            avg_reward = np.mean(reward_window)
            avg_steps = np.mean(step_window)
            avg_loss = np.mean(agent.losses[-100:]) if agent.losses else 0.0

            if episode % 20 == 0:  #every 20 episodes
                print("\nRunning evaluation episode (greedy policy)...")
                eval_reward, eval_steps = run_eval_episode(env, agent)
                print(f"  Eval reward: {eval_reward:.2f} over {eval_steps} steps\n")

            # Log to CSV
            with open(LOG_CSV, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode, total_reward, avg_reward, 
                               agent.epsilon, agent.beta, steps, avg_loss])

            # Console output
            print(f"Episode {episode:4d} | "
                  f"Reward: {total_reward:7.2f} | "
                  f"Avg(100): {avg_reward:7.2f} | "
                  f"Steps: {steps:4d} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"β: {agent.beta:.3f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Time: {episode_time:.1f}s")

            # Save checkpoint
            if episode % SAVE_EVERY == 0:
                ckpt_path = agent.save(name=f"dqn_ep{episode}.pth")
                print(f"  → Saved checkpoint: {ckpt_path}")
                print(f"  → Buffer size: {len(agent.buffer)}")
                print(f"  → Avg reward (100 eps): {avg_reward:.2f}")
                print(f"  → Avg steps (100 eps): {avg_steps:.1f}")

            # Periodic performance check
            if episode % 100 == 0 and episode > 0:
                print("\n" + "=" * 60)
                print(f"CHECKPOINT at Episode {episode}")
                print(f"  Average Reward (last 100): {avg_reward:.2f}")
                print(f"  Average Steps (last 100): {avg_steps:.1f}")
                print(f"  Epsilon: {agent.epsilon:.3f}")
                print(f"  Total training steps: {global_step}")
                print("=" * 60 + "\n")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        ckpt_path = agent.save(name="dqn_interrupt.pth")
        print(f"Saved interrupt checkpoint: {ckpt_path}")
    
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        ckpt_path = agent.save(name="dqn_error.pth")
        print(f"Saved error checkpoint: {ckpt_path}")
    
    finally:
        env.close()
        print("\n" + "=" * 60)
        print("TRAINING FINISHED")
        print(f"Final checkpoint saved")
        print(f"Total episodes completed: {episode}")
        print("=" * 60)


if __name__ == "__main__":
    train()