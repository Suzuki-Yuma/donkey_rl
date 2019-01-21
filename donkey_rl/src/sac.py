import gym
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
from collections import deque
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import visdom

import donkey_gym
import my_cv

EPISODES = 1000
img_rows , img_cols = 80, 80
# Convert image into Black and white
img_channels = 4 # We stack 4 frames

class TanGaussianPolicy(nn.Module):
    def __init__(self, action_dim):
        super(TanGaussianPolicy, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=8, stride=4, padding=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(10*10*64, 512)
        self.linear_for_mu = nn.Linear(512, action_dim)
        self.linear_for_sigma = nn.Linear(512, action_dim)

        init_w = 1e-3
        self.linear_for_mu.weight.data.uniform_(-init_w, init_w)
        self.linear_for_mu.bias.data.uniform_(-init_w, init_w)
        self.linear_for_sigma.weight.data.uniform_(-init_w, init_w)
        self.linear_for_sigma.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        h = F.relu(self.conv1(state))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.view(-1, 10*10*64)
        h = F.relu(self.linear1(h))
        mu = self.linear_for_mu(h)
        log_sigma = self.linear_for_sigma(h)
        log_sigma = torch.clamp(log_sigma, -20, 2)
        sigma = torch.exp(log_sigma)

        normal = Normal(mu, sigma)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action * action + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob, mu, sigma

class Qfunc(nn.Module):
    def __init__(self, action_dim):
        super(Qfunc, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=8, stride=4, padding=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(10*10*64, 512)
        self.action_emb = nn.Linear(action_dim, 128)
        self.linear2 = nn.Linear(512+128, 512)
        self.linear3  = nn.Linear(512, 1)

    def forward(self, state, action):
        h = F.relu(self.conv1(state))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.view(-1, 10*10*64)
        h = F.relu(self.linear1(h))
        h = F.relu(self.linear2(torch.cat((h, F.relu(self.action_emb(action))), dim=1)))
        h = self.linear3(h)
        return h

class Vfunc(nn.Module):
    def __init__(self):
        super(Vfunc, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=8, stride=4, padding=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(10*10*64, 512)
        self.linear2 = nn.Linear(512, 1)

    def forward(self, state):
        h = F.relu(self.conv1(state))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.view(-1, 10*10*64)
        h = F.relu(self.linear1(h))
        h = self.linear2(h)
        return h

class SACAgent:

    def __init__(self, state_size, action_size):
        self.t = 0
        self.train = False
        self.load = True

        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.batch_size = 64
        self.train_start = 1000
        self.tau = 0.001
        self.reward_scale = 5.

        self.memory = deque(maxlen=50000)

        self.policy = TanGaussianPolicy(action_dim=action_size)
        self.qf = Qfunc(action_dim=action_size)
        self.target_qf = copy.deepcopy(self.qf)
        self.qf_criterion = nn.MSELoss()

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=3e-4,
        )
        self.qf_optimizer = torch.optim.Adam(
            self.qf.parameters(),
            lr=3e-4,
        )

        self.policy_losses = [0]
        self.qf_losses = [0]
        self.log_pis = [0]


    def process_image(self, obs):
        obs = skimage.color.rgb2gray(obs)
        obs = skimage.transform.resize(obs, (img_rows, img_cols))
        return obs


    def update_target_model(self):
        for target, source in zip(self.target_qf.parameters(), self.qf.parameters()):
            target.data.copy_(target.data * (1. - self.tau) + source * self.tau)

    def get_action(self, s_t):
        s_t_torch = torch.tensor(s_t)
        action, _, mu, sigma = self.policy(s_t_torch)
        return action.detach().numpy(), mu.detach().numpy(), sigma.detach().numpy()

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
        new_action, log_pi, mu, sigma = self.policy(state_t)
        new_action_t1, log_pi_t1, _, _ = self.policy(state_t1)

        target_v_value = self.target_qf(state_t1, new_action_t1) - log_pi_t1
        q_target = reward_t * self.reward_scale + (1. - terminal) * self.discount_factor * target_v_value
        qf_loss = self.qf_criterion(q_pred, q_target.detach())

        q_new_action = self.qf(state_t, new_action)
        policy_loss = (log_pi - q_new_action).mean() + 1e-3 * ((sigma**2).mean())

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.update_target_model()

        self.policy_losses.append(policy_loss.detach().item())
        self.log_pis.append(log_pi.mean().item())
        self.qf_losses.append(qf_loss.detach().item())

if __name__ == "__main__":
    env = gym.make("donkey-generated-roads-v0")

    # Get size of state and action from environment
    state_size = (img_rows, img_cols, img_channels)
    action_size = env.action_space.low.size # Steering and Throttle

    agent = SACAgent(state_size, action_size)

    vis = visdom.Visdom()

    episodes = []
    mus = [0]
    sigmas = [0]
    episode_lens = [0]
    returns = [0]

    mu_plot = vis.line(X=[-1], Y=mus, opts=dict(legend=["mu"]))
    sigma_plot = vis.line(X=[-1], Y=sigmas, opts=dict(legend=["sigma"]))
    episode_lens_plot = vis.line(X=[-1], Y=episode_lens, opts=dict(legend=["Episode Lengths"]))
    return_plot = vis.line(X=[-1], Y=returns, opts=dict(legend=["Return"]))
    policy_plot = vis.line(X=[-1], Y=agent.policy_losses, opts=dict(legend=["Policy Loss"]))
    log_pi_plot = vis.line(X=[-1], Y=agent.log_pis, opts=dict(legend=["Log Pi Mean"]))
    qf_plot = vis.line(X=[-1], Y=agent.qf_losses, opts=dict(legend=["Q function Loss"]))

    if agent.load:
        print("Now we load the saved model")
        agent.policy.load_state_dict(torch.load("./save_model/sac/sac_policy.pt"))
        agent.qf.load_state_dict(torch.load("./save_model/sac/sac_qf.pt"))


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
            if agent.train:
                action, mu, sigma = agent.get_action(s_t)
                mus.append(mu.mean())
                sigmas.append(sigma.mean())
            else:
                with torch.no_grad():
                    temp_s_t = torch.tensor(s_t).repeat(100, 1, 1, 1)
                    actions, _, _, _ = agent.policy(temp_s_t)
                    q_values = agent.qf(temp_s_t, actions)
                    action = actions[torch.argmax(q_values)].numpy()

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

                episodes.append(e)

                episode_lens.append(episode_len)
                returns.append(total_reward)

                vis.line(X=list(range(-1, len(mus)-1)), Y=mus, opts=dict(legend=["mu"]), win=mu_plot, update='replace')
                vis.line(X=list(range(-1, len(sigmas)-1)), Y=sigmas, opts=dict(legend=["sigma"]), win=sigma_plot, update='replace')
                vis.line(X=list(range(-1, len(episode_lens)-1)), Y=episode_lens, opts=dict(legend=["Episode Lengths"]), win=episode_lens_plot, update='replace')
                vis.line(X=list(range(-1, len(returns)-1)), Y=returns, opts=dict(legend=["Return"]), win=return_plot, update='replace')
                vis.line(X=list(range(-1, len(agent.policy_losses)-1)), Y=agent.policy_losses, opts=dict(legend=["Policy Loss"]), win=policy_plot, update='replace')
                vis.line(X=list(range(-1, len(agent.log_pis)-1)), Y=agent.log_pis, opts=dict(legend=["Log Pi Mean"]), win=log_pi_plot, update='replace')
                vis.line(X=list(range(-1, len(agent.qf_losses)-1)), Y=agent.qf_losses, opts=dict(legend=["Q function Loss"]), win=qf_plot, update='replace')

                # Save model for each episode
                if agent.train:
                    np.save("./save_model/sac/episode_len.npy", np.asarray(episode_lens))
                    np.save("./save_model/sac/returns.npy", np.asarray(returns))
                    torch.save(agent.policy.state_dict(), "./save_model/sac/sac_policy.pt")
                    torch.save(agent.qf.state_dict(), "./save_model/sac/sac_qf.pt")
                    torch.save(agent.target_qf.state_dict(), "./save_model/sac/sac_target_qf.pt")

                print("\nepisode:", e, "  memory length:", len(agent.memory),
                      " episode length:", episode_len, " Return:", total_reward)
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Return Curve by DDQN")
    plt.show()
