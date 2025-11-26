import csv
import matplotlib.pyplot as plt

episodes = []
rewards = []
avg_rewards = []
epsilons = []
losses = []

with open("training_log.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        episodes.append(int(row["episode"]))
        rewards.append(float(row["total_reward"]))
        avg_rewards.append(float(row["avg_reward_100"]))
        epsilons.append(float(row["epsilon"]))
        losses.append(float(row["avg_loss"]))

plt.figure()
plt.plot(episodes, rewards, label="episode reward")
plt.plot(episodes, avg_rewards, label="avg reward (100eps)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.show()

plt.figure()
plt.plot(episodes, epsilons)
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.show()

plt.figure()
plt.plot(episodes, losses)
plt.xlabel("Episode")
plt.ylabel("Avg loss (last 100)")
plt.show()
