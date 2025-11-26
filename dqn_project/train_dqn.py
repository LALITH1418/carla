# train_dqn.py - STABILIZED VERSION
from dqn_agent import DQNAgent
from carla_env import CarlaEnv
import time
import numpy as np
import csv
import os
from collections import deque

def train():
    # Initialize environment
    env = CarlaEnv()
    state_size = env.state_size
    action_size = env.action_space

    # Initialize agent with stabilized hyperparameters
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=1e-4,
        gamma=0.99,
        batch_size=64,
        buffer_size=100000,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=100000,
        update_every=4,          # Key: don't train every step
        target_update_freq=1000,
        tau=0.001,
        model_dir="models"
    )

    # Training parameters
    EPISODES = 1000
    SAVE_EVERY = 50
    LOG_CSV = "training_log.csv"
    WARMUP_STEPS = 1000  # Collect experiences before training
    
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
        global_step = 0
        
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