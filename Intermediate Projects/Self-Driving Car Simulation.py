"""
Self-Driving Car Simulation
Reinforcement Learning (DQN) + Rule-Based controller in a 2D environment.
Install: pip install numpy matplotlib pygame gymnasium torch
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from collections import deque
import random
import math

# ── 1. Environment ────────────────────────────────────────────────────────────
class SelfDrivingEnv:
    """
    2D top-down car simulation with a circular track.
    State  : [dist_left, dist_right, dist_front, speed, angle_to_track, curvature]
    Actions: 0=straight, 1=left, 2=right, 3=brake
    """
    WIDTH, HEIGHT = 800, 600
    TRACK_CX, TRACK_CY = 400, 300
    TRACK_R_OUTER = 250
    TRACK_R_INNER = 140

    def __init__(self):
        self.reset()

    def reset(self):
        # Start on the right side of the track midline
        mid_r = (self.TRACK_R_OUTER + self.TRACK_R_INNER) / 2
        self.car_angle = 0.0          # heading (radians)
        self.track_angle = 0.0        # position on track (radians)
        self.car_x = self.TRACK_CX + mid_r
        self.car_y = self.TRACK_CY
        self.speed = 3.0
        self.steer = 0.0
        self.lap_count = 0
        self.steps = 0
        self.total_reward = 0.0
        return self._get_state()

    # ── geometry helpers ──────────────────────────────────────────────────────
    def _dist_to_track_center(self):
        return math.hypot(self.car_x - self.TRACK_CX,
                          self.car_y - self.TRACK_CY)

    def _on_track(self):
        r = self._dist_to_track_center()
        return self.TRACK_R_INNER < r < self.TRACK_R_OUTER

    def _ray_distance(self, angle_offset: float, max_dist=200) -> float:
        """Cast a ray and return distance to nearest wall."""
        ray_angle = self.car_angle + angle_offset
        for d in range(1, max_dist + 1):
            x = self.car_x + d * math.cos(ray_angle)
            y = self.car_y + d * math.sin(ray_angle)
            r = math.hypot(x - self.TRACK_CX, y - self.TRACK_CY)
            if r > self.TRACK_R_OUTER or r < self.TRACK_R_INNER:
                return d / max_dist
        return 1.0

    def _get_state(self):
        r = self._dist_to_track_center()
        mid_r = (self.TRACK_R_OUTER + self.TRACK_R_INNER) / 2
        center_angle = math.atan2(self.car_y - self.TRACK_CY,
                                  self.car_x - self.TRACK_CX)
        tangent_angle = center_angle + math.pi / 2  # tangent to track
        angle_err = (self.car_angle - tangent_angle + math.pi) % (2*math.pi) - math.pi

        return np.array([
            self._ray_distance(-math.pi/4),  # front-left
            self._ray_distance(0),            # front
            self._ray_distance(math.pi/4),   # front-right
            self.speed / 10.0,
            angle_err / math.pi,
            (r - mid_r) / ((self.TRACK_R_OUTER - self.TRACK_R_INNER) / 2),
        ], dtype=np.float32)

    def step(self, action: int):
        STEER_SPEED, STEER_DECAY = 0.04, 0.9
        if   action == 1: self.steer -= STEER_SPEED
        elif action == 2: self.steer += STEER_SPEED
        elif action == 3: self.speed  = max(1.0, self.speed - 0.5)
        self.steer *= STEER_DECAY

        self.car_angle += self.steer
        self.speed = np.clip(self.speed + 0.05, 1.0, 8.0)
        self.car_x += self.speed * math.cos(self.car_angle)
        self.car_y += self.speed * math.sin(self.car_angle)

        prev_angle = self.track_angle
        self.track_angle = math.atan2(self.car_y - self.TRACK_CY,
                                      self.car_x - self.TRACK_CX)
        self.steps += 1

        on_track = self._on_track()
        reward   = 1.0 if on_track else -50.0
        done     = (not on_track) or self.steps > 2000

        if done and on_track:
            self.lap_count += 1
            reward += 200

        self.total_reward += reward
        return self._get_state(), reward, done, {'laps': self.lap_count, 'on_track': on_track}


# ── 2. DQN Agent ──────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, state_size=6, action_size=4):
        self.state_size  = state_size
        self.action_size = action_size
        self.memory      = deque(maxlen=50_000)
        self.gamma       = 0.99
        self.epsilon     = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.lr          = 1e-3
        self.batch_size  = 64
        self._build_model()

    def _build_model(self):
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim

            class QNet(nn.Module):
                def __init__(self, in_dim, out_dim):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(in_dim, 128), nn.ReLU(),
                        nn.Linear(128, 128),    nn.ReLU(),
                        nn.Linear(128, 64),     nn.ReLU(),
                        nn.Linear(64, out_dim)
                    )
                def forward(self, x): return self.net(x)

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model  = QNet(self.state_size, self.action_size).to(self.device)
            self.target = QNet(self.state_size, self.action_size).to(self.device)
            self.target.load_state_dict(self.model.state_dict())
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.loss_fn   = nn.MSELoss()
            self.USE_TORCH = True
            print(f"DQN using: {self.device}")

        except ImportError:
            self.USE_TORCH = False
            # Lightweight numpy Q-table approximation
            self.weights = np.random.randn(self.action_size, self.state_size) * 0.01
            print("PyTorch not found — using linear Q-approx")

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        if self.USE_TORCH:
            import torch
            with torch.no_grad():
                t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.model(t).argmax().item()
        return int(np.dot(self.weights, state).argmax())

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        if self.USE_TORCH:
            import torch
            states  = torch.FloatTensor([e[0] for e in batch]).to(self.device)
            actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
            rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
            next_s  = torch.FloatTensor([e[3] for e in batch]).to(self.device)
            dones   = torch.BoolTensor([e[4] for e in batch]).to(self.device)

            q_vals    = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
            with torch.no_grad():
                q_next = self.target(next_s).max(1)[0]
                q_next[dones] = 0.0
            targets = rewards + self.gamma * q_next

            loss = self.loss_fn(q_vals, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            for s, a, r, s2, done in batch:
                target = r if done else r + self.gamma * np.dot(self.weights, s2).max()
                self.weights[a] += self.lr * (target - np.dot(self.weights[a], s)) * s

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        if self.USE_TORCH:
            self.target.load_state_dict(self.model.state_dict())


# ── 3. Rule-Based controller (baseline) ───────────────────────────────────────
def rule_based_action(state: np.ndarray) -> int:
    dist_left, dist_front, dist_right, speed, angle_err, track_offset = state
    if dist_front < 0.3:
        return 3  # brake
    if angle_err > 0.1 or track_offset > 0.3:
        return 1  # steer left
    if angle_err < -0.1 or track_offset < -0.3:
        return 2  # steer right
    return 0  # straight


# ── 4. Training Loop ──────────────────────────────────────────────────────────
def train(episodes=200, max_steps=1000):
    env   = SelfDrivingEnv()
    agent = DQNAgent()

    rewards_history = []
    steps_history   = []
    epsilon_history = []

    print(f"\n{'─'*50}")
    print(f"{'Episode':>10} {'Reward':>10} {'Steps':>8} {'Laps':>6} {'ε':>8}")
    print(f"{'─'*50}")

    for ep in range(1, episodes + 1):
        state = env.reset()
        ep_reward, ep_steps = 0.0, 0

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            ep_reward += reward
            ep_steps   = step + 1
            if done:
                break

        if ep % 10 == 0:
            agent.update_target()

        rewards_history.append(ep_reward)
        steps_history.append(ep_steps)
        epsilon_history.append(agent.epsilon)

        if ep % 20 == 0 or ep <= 5:
            avg = np.mean(rewards_history[-20:])
            print(f"{ep:>10} {ep_reward:>10.1f} {ep_steps:>8} "
                  f"{info['laps']:>6} {agent.epsilon:>8.3f}  (avg20={avg:.1f})")

    return agent, rewards_history, steps_history, epsilon_history


# ── 5. Run Training ───────────────────────────────────────────────────────────
print("🚗 Starting Self-Driving Car Simulation…")
agent, rewards, steps, epsilons = train(episodes=150)

# ── 6. Visualizations ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Self-Driving Car DQN Training', fontsize=15, fontweight='bold')

# Track visualization
ax_track = axes[0, 0]
theta = np.linspace(0, 2*np.pi, 300)
env   = SelfDrivingEnv()
ax_track.plot(env.TRACK_CX + env.TRACK_R_OUTER * np.cos(theta),
              env.TRACK_CY + env.TRACK_R_OUTER * np.sin(theta), 'k-', lw=2)
ax_track.plot(env.TRACK_CX + env.TRACK_R_INNER * np.cos(theta),
              env.TRACK_CY + env.TRACK_R_INNER * np.sin(theta), 'k-', lw=2)
ax_track.fill_between(env.TRACK_CX + env.TRACK_R_OUTER * np.cos(theta),
                      env.TRACK_CY + env.TRACK_R_OUTER * np.sin(theta), color='gray', alpha=0.2)

# Simulate final agent trajectory
state = env.reset()
traj_x, traj_y = [env.car_x], [env.car_y]
for _ in range(800):
    action = agent.act(state)
    state, _, done, _ = env.step(action)
    traj_x.append(env.car_x); traj_y.append(env.car_y)
    if done: break
ax_track.plot(traj_x, traj_y, 'r-', lw=1.5, alpha=0.8, label='Agent path')
ax_track.plot(traj_x[0], traj_y[0], 'go', markersize=10, label='Start')
ax_track.set_title('Track & Agent Trajectory'); ax_track.legend()
ax_track.set_aspect('equal'); ax_track.axis('off')

# Reward over time
window = 10
smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
axes[0,1].plot(rewards, alpha=0.3, color='steelblue', label='Raw')
axes[0,1].plot(range(window-1, len(rewards)), smoothed, 'r-', lw=2, label=f'MA-{window}')
axes[0,1].set_title('Episode Reward'); axes[0,1].set_xlabel('Episode'); axes[0,1].legend()

# Steps per episode
axes[1,0].plot(steps, color='teal', alpha=0.7)
axes[1,0].set_title('Steps Survived per Episode'); axes[1,0].set_xlabel('Episode')

# Epsilon decay
axes[1,1].plot(epsilons, color='orange')
axes[1,1].set_title('Exploration Rate (ε)'); axes[1,1].set_xlabel('Episode')
axes[1,1].set_ylabel('ε'); axes[1,1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('self_driving_results.png', dpi=150)
print("📊 Saved: self_driving_results.png")
print(f"✅ Training complete! Best reward: {max(rewards):.1f} | Final ε: {epsilons[-1]:.3f}")
