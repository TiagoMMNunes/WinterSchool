'''
File used to plot the rewards and losses taken from the logs made by the DQN Agent
Author: Ir. Tiago Nunes
'''


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import numpy as np
import torch
from torch import optim
import random
from functools import reduce
import torch.nn as nn
from torch.autograd import Variable
import datetime
from pathlib import Path
from main import DQN_agent
from main import Net
from main import Replay_Buffer
from PIL import Image, ImageOps
import gym
import time
env = gym.make('ALE/Pong-v5')

# Initialize agent
net_output_size = env.action_space.n
Agent = DQN_agent(n_actions = net_output_size)

# Print nr params of network
print(sum(p.numel() for p in Agent.net.parameters()))

# Load agent
Agent.load_checkpoint()

# Get loss from agent and numpy it
Jtd_loss = torch.Tensor(Agent.Jtd_storage)
Jtd_loss = Jtd_loss.cpu()
Jtd_loss = Jtd_loss.detach().numpy()

# Moving average over 10
n = 100
mv = []
length = len(Jtd_loss)
j = 0
print(length)
for i in range(length):
    if n < i < length-n:
        suma = sum((Jtd_loss[0+i:i+n]))/n
        mv.append(suma)

# Plot loss, last 2000 and first 2000 samples of loss
plt.figure()
plt.plot(mv)
plt.title('DQN Learning loss')
plt.ylabel('Total Loss')
plt.xlabel('No. of samples')
plt.grid()
plt.show(block=False)

plt.figure()
plt.plot(mv[-2000:])
plt.title('DQN Learning loss, last N iterations')
plt.ylabel('Total Loss')
plt.xlabel('No. of samples')
plt.grid()
plt.show(block=False)

plt.figure()
plt.plot(mv[0:2000])
plt.title('DQN Learning loss, first N iterations')
plt.ylabel('Total Loss')
plt.xlabel('No. of samples')
plt.grid()
plt.show(block=False)

# Reward Plotting
rewards = Agent.rewards
mv_rewards=[]
mv = []
for i in range(len(rewards)):
    if n < i < length -n:
        suma_rewards = sum((rewards[0+i:i+n]))/n
        mv_rewards.append(suma_rewards)

plt.figure()
plt.plot(mv_rewards)
plt.title('DQN Learning rewards')
plt.ylabel('Rewards')
plt.xlabel('No. of samples')
plt.grid()
plt.show(block=False)

# Plot iterations per episode
n = 10
iters_per_episode = Agent.iters_per_episode
sum_iters_per_episode = []
for i in range(len(iters_per_episode)):
    if n < i < length - n:
        suma = float(sum((iters_per_episode[0 + i:i + n]))) / n
        sum_iters_per_episode.append(suma)
plt.figure()
plt.plot(sum_iters_per_episode)
plt.title('DQN iters per episode [training]')
plt.ylabel('Nr of iters [-]')
plt.xlabel('Nr of episodes')
plt.grid()
plt.show(block=False)





