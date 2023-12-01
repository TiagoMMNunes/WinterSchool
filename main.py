
import numpy as np
import cv2
from copy import deepcopy as dp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim import Adam
import sys
import time

# TODO: DQN_AGENT FUNCTION




# TODO: DQN_NET FUNCTION

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# Agent = DQFD_agent()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import gymnasium as gym
import time
# env = gym.make('CartPole-v0')
env = gym.make('CartPole-v1')

# observation = env.reset()
# action = env.action_space.sample()
#
# observation, reward, done, info = env.step(action)
# TODO: DQN Agent and Network

net_output_size = env.action_space.n
Agent = DQN_Agent(n_actions = net_output_size)
# Agent.load_checkpoint()
# Agent.net = Agent.net.float()
# Agent.target_net = Agent.target_net.float()
# print(np.shape(observation))
# gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
# # print(np.shape(gray))
# resized_image = cv2.resize(gray, (84, 84))
# # print(np.shape(resized_image))
# net_input_size = [84, 84]
# net_output_size = env.action_space.n
# # Recasting images from [0,255] to [0,1]
# net_input = torch.unsqueeze(torch.from_numpy(resized_image), dim = 0) / 255
# # net_input = torch.unsqueeze(torch.from_numpy(net_input), dim = 0)
# net_input = torch.div(net_input, 255)
# net_input = net_input[None, :]
# net_output = Net.forward(net_input)

# print(net_output)

# img = mpimg.imread('your_image.png')
# imgplot = plt.imshow(observation)
# plt.show()
# imgplot = plt.imshow(gray)
# plt.show()
# imgplot = plt.imshow(resized_image)
# plt.show()
replay_buffer_Full = False
curr_episode = -1
do_random_actions = True
max_episodes = 10
# Taking random actions
# By default, as a simplifying measure, demonstrations are collected as random actions using the command
# env.action_space.sample() command
while do_random_actions:
    episode_not_done = 1
    curr_episode += 1
    observation_old = env.reset()
    curr_episode_length = 0
    t = 0
    while episode_not_done:
        # env.render()
        # print(observation)

        action = env.action_space.sample()

        observation_new, reward, done, info = env.step(action)

        # Convert previous observation into Image (s)
        gray_old = cv2.cvtColor(observation_old, cv2.COLOR_BGR2GRAY)
        resized_image_old = cv2.resize(gray_old, (84, 84))

        # Convert new state into Image (s_)
        gray_new = cv2.cvtColor(observation_new, cv2.COLOR_BGR2GRAY)
        resized_image_new = cv2.resize(gray_new, (84, 84))

        # Transitions are stored as S, A, R, S_, done, is_demo, weight, idx of sample, curr_episode
        transition = [resized_image_old, action, reward, resized_image_new, done, True, demo_sample_weight\
                      , t, curr_episode]
        transition = np.array(transition)

        # print(action)
        # If the replay buffer is full then stop collecting demonstrations
        # if done:
        #     imgplot = plt.imshow(resized_image_new)
        #     plt.show()
        #     time.sleep(2)
        # Checks if episode has been concluded

        if done or t == 1000:
            episode_not_done = 0

        if curr_episode >= max_episodes:
            do_random_actions = False

        observation_old = observation_new
        t += 1
        curr_episode_length = dp(t)


env.close()



# Training cycle
env = gym.make('Alien-v0')
training_epochs = 0
max_epochs = 100000

start_time_train = time.time()
log_file = "logging_rewards_" + str(start_time_train) + ".txt"

Train = True
while Train:

    episode_not_done = 1
    curr_episode += 1
    observation_old = env.reset()
    curr_episode_length = 0
    curr_reward = 0
    t = 0

    while episode_not_done:
        # env.render()
        # print(observation)

        # Convert previous observation into Image (s)
        gray_old = cv2.cvtColor(observation_old, cv2.COLOR_BGR2GRAY)
        resized_image_old = cv2.resize(gray_old, (84, 84))

        # Convert image into batch of size 1 with proper sizing for net
        states_torch = torch.unsqueeze(torch.from_numpy(resized_image_old), dim=0)
        states_torch = torch.unsqueeze(states_torch, dim=0)
        states_torch = torch.div(states_torch, 255)

        action = Agent.act(states_torch)

        observation_new, reward, done, info = env.step(action)

        reward_agent = np.sign(reward) * np.log(1 + np.abs(reward))

        # Convert new state into Image (s_)
        gray_new = cv2.cvtColor(observation_new, cv2.COLOR_BGR2GRAY)
        resized_image_new = cv2.resize(gray_new, (84, 84))

        # Transitions are stored as S, A, R, S_, done, is_demo, weight, idx of sample, curr_episode
        transition = [resized_image_old, action, float(reward_agent), resized_image_new, done, False, training_sample_weight \
                      ,t, curr_episode]

        transition = np.array(transition)

        Agent.store_transition(transition)

        # if training_epochs % 100 == 0:
            # print(np.shape(Agent.memory.stored_transitions))
        Agent.update()
        # print(action)
        # If the replay buffer is full then stop collecting demonstrations
        # if done:
        #     imgplot = plt.imshow(resized_image_new)
        #     plt.show()
        #     time.sleep(2)
        # Checks if episode has been concluded

        if done: # or t == 1000:
            episode_not_done = 0

        observation_old = observation_new
        t += 1
        curr_episode_length = dp(t)
        training_epochs += 1
        curr_reward = curr_reward + reward

    # Storing length of current demo_episode
    # Agent.memory.demo_replay_episode_size.append(curr_episode_length)
    # Check if the max number of epochs have been reached
    if training_epochs > max_epochs:
        Train = False
    Agent.agent_returns.append(curr_reward)
    f = open(log_file, "a")
    f.write("Reward: " + str(curr_reward) + " episode length: " + str(t) + "\n")
    f.close()





#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
########## Bellow is leftover testing and legacy code not currently in use ############################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
# samples = test
# # next_state_values[non_final_mask] =  torch.from_numpy(Agent.calc_q_value(net_input))
# # next_state_values[non_final_mask] = Agent.calc_q_value(net_input)
# # print(next_state_values)
# non_final_mask = torch.tensor(tuple(map(lambda s: s is not True,
#                                       samples[:, 4])), dtype=torch.bool)
# next_state_values = torch.zeros(len(samples[:, 0]))
#
# # Converts the states and next_states into numpy arrays to be better used with torch
# states = np.array(list(samples[:, 0]), dtype=np.float32)
# next_states = np.array(list(samples[:, 3]), dtype=np.float32)
#
# # Converts numpy arrays into torch and divides them by 255 to cast them into [0,1].
# states_torch = torch.unsqueeze(torch.from_numpy(states), dim=1)
# states_torch = torch.div(states_torch, 255)
# # print(np.shape(net_input))
# # Applies the same logic to next_states
# # Only receives the next_states that correspond to the non_final ones, since the value of the next_state in case
# # the state is terminal is 0, so no need to calculate it
# next_states_torch = torch.unsqueeze(torch.from_numpy(next_states), dim=1)[non_final_mask]
# next_states_torch = torch.div(next_states_torch, 255)
#
# # Receives actions sorted into the best ones. Also receives all the q values
# # sortedA, net_output = self.calc_best_action(net_input)
# # print(sortedA[1,:])
# # Calculates the Q value of the best action per state
# state_values = torch.from_numpy(Agent.calc_best_q_value(states_torch))
# # print(np.shape(best_output))
#
# # Replaces the corresponding values of the next_state with the output of the network except for those states
# # that are terminal. Those remain = 0
# next_state_values[non_final_mask] =torch.from_numpy( Agent.calc_best_q_value(next_states_torch))
# # print(np.shape(next_state_values))
# # print(next_state_values)
#
# # Getting reward values and converting to torch
# rewards = np.array(list(samples[:, 2]), dtype=np.float32)
# rewards_torch = torch.from_numpy(rewards)
#
# # Calculates expected value based on reward and expected value of next state
# expected_state_values = (next_state_values * Agent.gamma) + rewards_torch
#
# errors = torch.abs(expected_state_values - state_values)
# print(errors)
# print(np.shape(errors))
# Defining priorities for the demos
# def _get_priority(self, error, demo):
#     return (np.abs(error + self.e_a + self.e_d) ** self.a) if demo else (np.abs(error + self.e_a) ** self.a)
# for t in range(2000):
#
#     Conduct pre-training
    # sample = Agent.memory.sample_replay_buffer()






# # Pre-Training Cycle
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         # env.render()
#         # print(observation)
#
#         action = env.action_space.sample()
#
#         observation, reward, done, info = env.step(action)
#         gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
#         # print(np.shape(gray))
#         resized_image = cv2.resize(gray, (84, 84))
#         # print(np.shape(resized_image))
#
#         net_input_size = [84, 84]
#
#
#         # Recasting images from [0,255] to [0,1]
#         net_input = torch.unsqueeze(torch.from_numpy(resized_image), dim=0)
#         net_input = torch.div(net_input, 255)
#         # Expanding to [1,1,img_size,img_size]
#         net_input = net_input[None, :]
#         net_output = Agent.net.forward(net_input)
#         net_q_values = Agent.calc_q_value(net_input)
#         net_sortedA= Agent.calc_best_action(net_input)
#
#         # print(np.shape(net_q_values))
#         print(np.shape(net_sortedA))
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()