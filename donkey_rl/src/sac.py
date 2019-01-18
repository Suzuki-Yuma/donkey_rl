import gym
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
from collections import deque
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import donkey_gym
import my_cv

EPISODES = 200
img_rows , img_cols = 80, 80
# Convert image into Black and white
img_channels = 4 # We stack 4 frames

class TanGaussianPolicy(nn.Module):
    def __init__(self, action_dim):
        super(TanGaussianPolicy, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.linear1 = nn.Linear(6*6*64, 512)
        self.linear_for_mu = nn.Linear(512, action_dim)
        self.linear_for_sigma = nn.Linear(512, action_dim)

    def forward(self, state):
        h = F.relu(self.conv1(state))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.view(-1, 6*6*64)
        h = F.relu(self.linear1(h))
        mu = self.linear_for_mu(h)
        log_sigma = self.linear_for_sigma(h)
        log_sigma = torch.clamp(log_sigma, -20, 2)
        sigma = torch.exp(log_sigma)

        normal = Normal(mu, sigma)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action * action + 1e-6)

        return action, log_prob

class Qfunc(nn.Module):
    def __init__(self, action_dim):
        super(Qfunc, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.linear1 = nn.Linear(6*6*64+action_dim, 512)
        self.linear2 = nn.Linear(512, 1)

    def forward(self, state, action):
        h = F.relu(self.conv1(state))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.view(-1, 6*6*64)
        h = F.relu(self.linear1(torch.cat((h, action), dim=1)))
        h = self.linear2(h)
        return h

class Vfunc(nn.Module):
    def __init__(self):
        super(Vfunc, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.linear1 = nn.Linear(6*6*64, 512)
        self.linear2 = nn.Linear(512, 1)

    def forward(self, state):
        h = F.relu(self.conv1(state))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.view(-1, 6*6*64)
        h = F.relu(self.linear1(h))
        h = self.linear2(h)
        return h

class SACAgent:

    def __init__(self, state_size, action_size):
        self.t = 0
        self.train = True
        self.load = False

        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.batch_size = 128
        self.train_start = 1000
        self.tau = 0.001

        self.memory = deque(maxlen=1000000)

        self.policy = TanGaussianPolicy(action_dim=action_size)
        self.qf = Qfunc(action_dim=action_size)
        self.vf = Vfunc()
        self.target_vf = copy.deepcopy(self.vf)
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=3e-4,
        )
        self.qf_optimizer = torch.optim.Adam(
            self.qf.parameters(),
            lr=3e-4,
        )
        self.vf_optimizer = torch.optim.Adam(
            self.vf.parameters(),
            lr=3e-4,
        )


    def process_image(self, obs):
        obs = skimage.color.rgb2gray(obs)
        obs = skimage.transform.resize(obs, (img_rows, img_cols))
        return obs


    def update_target_model(self):
        for target, source in zip(self.target_vf.parameters(), self.vf.parameters()):
            target.data.copy_(target.data * (1. - self.tau) + source * self.tau)

    def get_action(self, s_t):
        s_t_torch = torch.tensor(s_t)
        action, _ = self.policy(s_t_torch)
        return action.detach().numpy()

    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return

        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
        state_t = torch.tensor(np.concatenate(state_t))
        action_t = torch.tensor(np.concatenate(action_t))
        state_t1 = torch.tensor(np.concatenate(state_t1))
        reward_t = torch.tensor(np.asarray(reward_t).astype(np.float32))
        terminal = torch.tensor(np.asarray(terminal).astype(np.float32))

        q_pred = self.qf(state_t, action_t)
        v_pred = self.vf(state_t)
        new_action, log_pi = self.policy(state_t)

        target_v_value = self.target_vf(state_t1)
        q_target = reward_t + (1. - terminal) * self.discount_factor * target_v_value
        qf_loss = self.qf_criterion(q_pred, q_target.detach())

        q_new_action = self.qf(state_t, new_action)
        v_target = q_new_action - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())

        policy_loss = (log_pi - q_new_action).mean()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.update_target_model()

if __name__ == "__main__":
    env = gym.make("donkey-generated-roads-v0")

    # Get size of state and action from environment
    state_size = (img_rows, img_cols, img_channels)
    action_size = env.action_space.low.size # Steering and Throttle

    agent = SACAgent(state_size, action_size)

    episodes = []
    episode_lens = []
    returns = []

    if agent.load:
        print("Now we load the saved model")


    for e in range(EPISODES):

        print("Episode: ", e)

        done = False
        obs = env.reset()

        episode_len = 0
        total_reward = 0.

        x_t = agent.process_image(obs)

        s_t = np.stack((x_t,x_t,x_t,x_t),axis=0)
        s_t = s_t.reshape(1, 4, img_cols, img_rows).astype(np.float32)

        while not done:

            # Get action for the current state and go one step in environment
            action = agent.get_action(s_t)
            real_action = (action[0] + 1) / 2. * (env.action_space.high - env.action_space.low) + env.action_space.low
            next_obs, reward, done, info = env.step(real_action)

            x_t1 = agent.process_image(next_obs)

            x_t1 = x_t1.reshape(1, 1, img_cols, img_rows)

            s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1).astype(np.float32) #1x80x80x4

            # Save the sample <s, a, r, s'> to the replay memory
            agent.replay_memory(s_t, action, reward, s_t1, int(done))

            if agent.train:
                agent.train_replay()

            s_t = s_t1
            agent.t = agent.t + 1
            episode_len = episode_len + 1
            total_reward += reward
            print("\rEPISODE {0} TIMESTEP {1} / RETURN {2} / EPISODE LENGTH {3}".format(e, agent.t, total_reward, episode_len), end="")

            if done:

                # Every episode update the target model to be same with model
                agent.update_target_model()

                episodes.append(e)

                episode_lens.append(episode_len)
                returns.append(total_reward)

                # Save model for each episode
                if agent.train:
                    np.save("./save_model/sac/episode_len.npy", np.asarray(episode_lens))
                    np.save("./save_model/sac/returns.npy", np.asarray(returns))
                    torch.save(agent.policy.state_dict(), "./save_model/sac/sac_policy.pt")
                    torch.save(agent.qf.state_dict(), "./save_model/sac/sac_qf.pt")
                    torch.save(agent.vf.state_dict(), "./save_model/sac/sac_vf.pt")
                    torch.save(agent.target_vf.state_dict(), "./save_model/sac/sac_target_vf.pt")

                print("\nepisode:", e, "  memory length:", len(agent.memory),
                      " episode length:", episode_len, " Return:", total_reward)
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Return Curve by DDQN")
    plt.show()
