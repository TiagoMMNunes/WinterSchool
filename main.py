
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
import torch.optim as optim
import sys
import time
import math as mh
import random
from pathlib import Path


class Replay_Buffer():
    """Function implementing a typical DQN replay buffer
        Author:  Ir. Tiago Nunes, 2023"""
    def __init__(self, memory_size = 10000, batch_size = 128):
        #
        self.memory_size = memory_size

        # Start memory counter at the number of demos available, to be used for determining position of new transitions
        self.memory_counter = 0
        # To keep track of the amount of demonstrations currently in the buffer
        self.current_memory_size = 0
        # Batch size
        self.batch_size = batch_size

        self.stored_transitions = [None] * memory_size
        # Create function to start up the demos loading them
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def store_sample(self, transition):
        # Storage follows a FIFO approach

        memory_place = self.memory_counter % self.memory_size

        transition = np.append(transition, memory_place)

        self.stored_transitions[memory_place] = transition

        # if self.memory_counter < self.memory_size:
        #     self.stored_transitions.append(transition)
        #
        # if self.memory_counter >= self.memory_size:
        #     self.stored_transitions[memory_place] = transition

        self.memory_counter += 1
        self.current_memory_size += 1


    # Currently applying uniform sampling.
    # Will change to prioritized sampling later on
    def sample_replay_buffer(self, batch_size=128):
        useful_mem_size = min(self.memory_counter, self.memory_size)
        numpy_transitions = np.array(self.stored_transitions[0:useful_mem_size-1])

        # New implementation
        my_generator = np.random.default_rng()
        # print(p.sum())
        sample = my_generator.choice(numpy_transitions,size=batch_size, replace=False)


        return sample

    def update_prioritization_weights(self, td_errors, samples, beta):

        return torch.ones(np.shape(td_errors)).cuda()
# Transitions are stored as S, A, R, S_, done, is_demo, weight, idx of sample, curr_episode, memory_place



class Net(nn.Module):
    """Function implementing a convulutional neural network with several layers
            Author:  Ir. Tiago Nunes, 2023"""
    def __init__(self, fin = 4, fout = 18, img_size = 84):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.fout = fout
        self.fin = fin

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8,stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # self.max1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(16384, 512)
        self.fc2 = nn.Linear(512,fout)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))
        # x = self.max1(x)
        x = torch.flatten(x, start_dim =  1)
        # print(np.shape(x))
        # x = torch.flatten(x, start_dim =  0)

        # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        # print(np.shape(self.fc2(x)))
        return self.fc2(x)

class DQN_agent():
    """Function implementing a DQN agent.
                Author:  Ir. Tiago Nunes, 2023"""
    def __init__(self,eps = 0.8, eps_decay = 0.05, lr = 1e-3,gamma = 0.99, mbsize = 128, C = 500, N = 10000,
                 n_states = 4, n_actions = 2, a = 0.4,
                 training_length = 100000, net_output_size = 6
                 , possible_action_values=[-50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0\
                          , 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]):
        self.possible_action_values = possible_action_values
        self.eps = eps # Epsilon greedy
        self.eps_decay = eps_decay
        self.lr = lr # Learnign rate
        self.gamma = gamma # Ga,,a
        self.mbsize = mbsize # Mini batch size
        self.target_C = C # Frequency of target updating
        self.target_c = 0 # Counter for target updating
        self.memory = Replay_Buffer(memory_size=N) # Memory size

        # Lambda losses. values equal to original paper
        # Below are the ea/ed values from the paper
        # self.ed = 0.005  # bonus for demonstration
        # self.ea = 0.001
        self.n_states = n_states
        self.n_actions = net_output_size
        self.net = Net(fin = n_states, fout = net_output_size).cuda()

        self.target_net = Net(fin = n_states, fout = net_output_size).cuda()
        self.target_net.load_state_dict(self.net.state_dict())

        # self.opt = optim.Adam(self.net.parameters(), lr = self.lr, weight_decay = self.lambda3)
        # self.opt = optim.Adam(self.net.parameters(), weight_decay=1e-5, lr=self.lr,
        #                       betas=(0.9, 0.999), amsgrad=True)
        # Defining action set index
        self.A = np.array([i for i in range(n_actions)])
        # Loss storage variables
        self.Jn_storage = []
        self.Je_storage = []
        self.Jtd_storage = []
        self.Loss_storage = []

        # Reward and sucess variables
        self.rewards = []
        self.success = []
        self.episode_size = []

        # Number of nonzero actions
        self.non_zero_train = []
        self.non_zero_eval = []

        # Nr iters per episode
        self.iters_per_episode = []

        # Did the agent successfully reach the target(1) or not(0)?
        self.reached_target = []

        # Evaluation variables for continuous training
        self.evals_iters_per_episode = []
        self.evals_reached_target = []

        # Evaluation variables
        self.eval_rewards = []
        self.eval_successes = []
        self.eval_actions = []

        # Actions taken per episode
        self.train_actions_cont = []
        self.eval_actions_cont = []

        # Perfect actions
        self.perfect_actions = []

        # Counters for training
        self.pre_train_step_number = 0
        self.training_step_number = 0

        # Reporting frequency for console updates for progress
        self.report_frequency = 100
        self.train_length = training_length
        self.start_time_pre_train = time.time()

        self.log_file = "logging_losses_" + str(self.start_time_pre_train) + ".txt"

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss = nn.MSELoss(reduction='none')

        self.net.to(self.device)
        self.opt = optim.Adam(self.net.parameters(), weight_decay=1e-5, lr=self.lr,
                              betas=(0.9, 0.999), amsgrad=True)

        # Possible actions
        self.possible_actions = list(range(0, self.n_actions))
        self.saving_path_full = ".\December04.pt"
        # For loading
        self.checkpoint_path_full = ".\December04.pt"

        import platform
        if platform.system() == "Linux" or platform.system() == "MacOS" :
            self.saving_path_full = self.saving_path_full.replace("\\", "/")
            self.checkpoint_path_full = self.checkpoint_path_full.replace("\\", "/")

    def save_checkpoint(self):

        # if self.state != None and self.epoch == 1:
        #    self.writer.add_graph(self.policy_net, self.state)
        saving_path = self.saving_path_full
        print(f'Saving model as {saving_path} at epoch #{self.target_c}')

        torch.save(
            {
                'epoch': self.target_c,
                'episode': self.target_c,
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.opt.state_dict(),
                'JE_losses_epochs': self.Je_storage,
                'double_DQN_losses_epochs': self.Jtd_storage,
                'n_step_losses_epochs': self.Jn_storage,
                'total_losses': self.Loss_storage,
                'rewards': self.rewards,
                'successes': self.success,
                'eps': self.eps,
                'eval_rewards': self.eval_rewards,
                'eval_successes': self.eval_successes,
                'eval_actions': self.eval_actions,
                'eval_iters_per_episode': self.evals_iters_per_episode,
                'evals_reached_target': self.evals_reached_target,
                'iters_per_episode': self.iters_per_episode,
                'reached_target': self.reached_target,
                # Actions for continuous agents
                'train_actions_cont': self.train_actions_cont,
                'eval_actions_cont': self.eval_actions_cont,
                'training_step_number':self.training_step_number,
                'perfect_actions': self.perfect_actions,
                'non_zero_train': self.non_zero_train,
                'non_zero_eval': self.non_zero_eval,
                # 'smoothed_episode_rewards': self.smoothed_episode_rewards,
                # 'total_frames_updates_episodes': self.total_frames_updates_episodes,
                # 'total_frames_updates_epochs': self.total_frames_updates_epochs,
                # 'frames_acquired_list_episodes': self.frames_acquired_list_episodes,
                # 'frames_acquired_list_epochs': self.frames_acquired_list_epochs,
                # 'nonzero_actions_episodes': self.nonzero_actions_episodes,
                'memory': self.memory,
                # 'pretrain_steps_performed': self.pretrain_steps_performed,
                # 'dict_hyperparameters': self.dict_hyperparameters,
                # 'conflict_list': self.conflict_list,
                # 'reached_target_list': self.reached_target_list,
                # 'total_reward_target_list': self.total_reward_target_list,
                # 'total_reward_nonzero_action_list': self.total_reward_nonzero_action_list,
                # 'total_reward_short_term_conflict_list': self.total_reward_short_term_conflict_list,
                # 'total_reward_medium_term_conflict_list': self.total_reward_medium_term_conflict_list,
                # 'total_reward_reached_target_list': self.total_reward_reached_target_list,
                # 'total_reward_flight_path_list': self.total_reward_flight_path_list,
                # 'demos_ratio_epochs': self.demos_ratio_epochs,
            },
            saving_path
        )

    def load_checkpoint(self):
        print(self.checkpoint_path_full)

        if Path(self.checkpoint_path_full).is_file():
            print('Using checkpoint file...')

            # checkpoint = torch.load(self.checkpoint_path_full,map_location=lambda storage, loc: storage)
            # Can use GPU whenever you have a large enough VRAM, not the case in this laptop
            use_gpu = False
            if self.device == torch.device('cuda:0') and use_gpu:
            # if False:
                print("Using GPU to load checkpoint\nCheck if you have enough VRAM or might be booted by OS")
                checkpoint = torch.load(self.checkpoint_path_full)
            else:
                print("Using CPU to load checkpoint")
                checkpoint = torch.load(self.checkpoint_path_full, map_location='cpu')
                print("Checkpoint loaded")

            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.target_net.load_state_dict(self.net.state_dict())


            self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
            self.target_c = checkpoint['epoch']
            self.training_step_number = checkpoint['training_step_number']

            self.Je_storage = checkpoint['JE_losses_epochs']
            self.Jtd_storage = checkpoint['double_DQN_losses_epochs']
            self.Jn_storage = checkpoint['n_step_losses_epochs']
            self.Loss_storage = checkpoint['total_losses']

            self.rewards = checkpoint['rewards']
            self.success = checkpoint['successes']

            try:
                self.non_zero_train = checkpoint['non_zero_train']
                self.non_zero_eval = checkpoint['non_zero_eval']

            except:
                print("No non zero actions")

            try:
                self.iters_per_episode = checkpoint['iters_per_episode']
                self.reached_target = checkpoint['reached_target']
                self.train_actions_cont = checkpoint['train_actions_cont']

            except:
                print("No continuous run training variables found")

            try:
                self.evals_iters_per_episode = checkpoint['eval_iters_per_episode']
                self.evals_reached_target = checkpoint['evals_reached_target']
                self.eval_actions_cont = checkpoint['eval_actions_cont']

            except:
                print("No continuous run evaluation variables found")

            try:
                self.perfect_actions = checkpoint['perfect_actions']
            except:
                print("No perfect actions found")

            self.eval_successes = checkpoint['eval_successes']
            self.eval_rewards = checkpoint['eval_rewards']
            self.eval_actions = checkpoint['eval_actions']

            self.eps = checkpoint['eps']
            self.memory = checkpoint['memory']
            self.beta = checkpoint['beta']

            # Decomposed Rewards
            self.target_net.eval()

            self.training_step_number = checkpoint['training_step_number']
            self.pre_train = checkpoint['pre_train']

            print("Checkpoint Load successful")

            try:
                self.pre_train_length = checkpoint['pre_train_length']
            except:
                print("No pre train length found")
        else:
            print('No checkpoint file found, train from random initialisation...')

    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()

    def act(self, state):
        # Epsilon_greedy actions
        # Generating random number
        random_sample = random.random()
        # Decay epsilon every 300 epochs
        if self.training_step_number > 0 and self.training_step_number % 5000 == 0 and self.eps > 0.1:
            self.eps = self.eps - self.eps_decay

        # Barrier against oversubtracting eps value
        if self.eps < 0.1:
            self.eps = 0.1

        if random_sample > self.eps:
            # with torch.nograd():
            # New way of doing act selection
            self.net.eval()
            state = state.to(self.device)
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.

                # action = self.net(state).max(1)[1].view(1, 1)
                # action = self.net(state).max(1)[1].view(1, 1)
                action = torch.argmax( self.net(state))

            self.net.train()
            action = action.item()
            return action

            # print("GREEDY ACTION TAKEN")
            # Old way of doing it
            # sortedA, qvalues = self.calc_best_action(state)
            # action = sortedA[0,0]
            # print( "Greedy action! ")

        else:
            action = random.choice(self.possible_actions)
            # print("Random action! ")
            # print(action)
        return action

    def greedy_action(self, state):
        # Returns the value of the greedy action
        self.net.eval()
        state = state.to(self.device)
        with torch.no_grad():
            action = torch.argmax( self.net(state))

        self.net.train()
        action = action.item()
        # sortedA, qvalues = self.calc_best_action(state)
        # print(sortedA)
        # print(qvalues)
        return action

    def td_loss(self, samples):
        # Calculate which states are final based on the isdone flag. Mask returns true if state is not final, false
        # if state is final
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not True,
                                                samples[:, 4])), dtype=torch.bool)

        # Initialize values of the next states as 0
        next_state_values = torch.zeros(len(samples[:, 0]))
        next_state_values = next_state_values.to(self.device)

        # Converts the states and next_states into numpy arrays to be better used with torch
        states = np.array(list(samples[:, 0]), dtype=np.float32)
        next_states = np.array(list(samples[:, 3]), dtype=np.float32)

        # Converts numpy arrays into torch and divides them by 255 to cast them into [0,1].
        # states_torch = torch.unsqueeze(torch.from_numpy(states), dim=1)

        # states = np.moveaxis(states, -1, 1)

        # next_states = np.moveaxis(next_states, -1, 1)
        states_torch = torch.from_numpy(states)
        next_states_torch  = torch.from_numpy(next_states)

        # states_torch = torch.unsqueeze(states_torch, dim=0)
        states_torch = torch.reshape(states_torch, [-1, 4, 84,84]).cuda()
        # states_torch = states_torch.to(self.device)
        # states_torch = torch.div(states_torch, 255)

        next_states_torch_non_final = torch.from_numpy(next_states)[non_final_mask]
        # next_states_torch_non_final = torch.squeeze(next_states_torch_non_final, dim = 0)
        next_states_torch_non_final = torch.reshape(next_states_torch_non_final, [-1, 4, 84,84])
        next_states_torch_non_final = next_states_torch_non_final.to(self.device)

        self.target_net.eval()
        with torch.no_grad():
            # print("huda")
            # print(np.shape(states_torch))
            # print(np.shape(next_states_torch_non_final)) ddd
            # print(np.shape(self.target_net(next_states_torch_non_final)))
            next_state_values[non_final_mask] = self.target_net(next_states_torch_non_final).max(1)[0]

        actions_torch = torch.tensor(list(samples[:, 1]), device=self.device)
        rewards_torch = torch.tensor(list(samples[:, 2]), device=self.device)

        actions_torch2 = actions_torch
        actions_torch = torch.unsqueeze(actions_torch, dim=0)
        actions_torch2 = actions_torch2.unsqueeze(1)


        self.net.train()
        state_values2 = self.net(states_torch).gather(1, actions_torch2)

        rewards_torch = rewards_torch.float()
        # Calculates expected value based on reward and expected value of next state
        expected_state_values = (next_state_values.cuda() * self.gamma) + rewards_torch

        errors = torch.abs(state_values2 - expected_state_values.unsqueeze(1))

        return state_values2, expected_state_values.unsqueeze(1).float(), errors
    def double_dqn_loss(self, state_values, expected_state_values, weights):

        individual_losses = self.loss(state_values, expected_state_values)

        weights2 = weights
        weights = weights.unsqueeze(dim=0)
        weights2 = weights2.unsqueeze(1)
        new_update2 = individual_losses * weights2
        doubleq_loss = new_update2.mean()
        return doubleq_loss

    def update(self):
        # Reminder:
        # Transitions are stored as S, A, R, S_, done, is_demo, weight, idx of sample, curr_episode, idx_of_sample

        # Calculate loss
        # Calculate new params predict_net

        # Update net

        # K-steps update target net
        if Agent.memory.current_memory_size < self.mbsize * 2:
            return 0
        #
        # print("dsa")

        self.net.train()

        samples = np.array(self.memory.sample_replay_buffer())
        # samples = np.array(self.sample())

        state_values, expected_state_values, errors = self.td_loss(samples)

        # individual_loss = nn.MSELoss(reduction='none')(state_values, expected_state_values)
        # Getting IS_weights
        # errors = torch.tensor(errors, device=self.device)
        # TODO: Turn IS weights into identity matrix
        # self.memory.
        IS_weights = self.memory.update_prioritization_weights(errors.detach(), samples, 0)

        Jdouble_q_loss = self.double_dqn_loss(state_values, expected_state_values.detach(), IS_weights.detach())

        Jtd = Jdouble_q_loss

        J = Jdouble_q_loss
        self.opt.zero_grad()
        J.backward()
        torch.nn.utils.clip_grad_value_(self.net.parameters(), 100)
        self.opt.step()

        # Storing losses
        self.Jtd_storage.append(Jtd.detach().cpu())
        self.Loss_storage.append(J.detach().cpu())

        # update target net every C/2 updates of the "predict net"
        if self.target_c > 0 and self.target_c % self.target_C == 0:
            self.update_target_net()
            print("\n Target network has been updated \n")

            if self.target_c > 0 and self.target_c % (6 * self.target_C) == 0:
                self.net.cpu()
                # self.target_net.cpu()
                # Also save current networks.
                self.save_checkpoint()
                self.net.cuda()
                # self.target_net.cuda()

        self.target_c += 1

        # Updating pre_train/ Train counters

        end = time.time()
        time_consumed = end - self.start_time_pre_train
        self.training_step_number += 1
        if self.training_step_number % self.report_frequency == 0:
            print("Current on episode: " + str(self.training_step_number) + " of " + str(self.train_length) \
                  + " loss: " + str(J) + " elapsed time: " + str(time_consumed) + " Current eps:" + str(self.eps))

        del Jdouble_q_loss
        del Jtd
        del J
        del IS_weights
        del state_values
        del expected_state_values
        del errors

        # TO VERIFY: Should MSE be considering the prioritization? The website version considers a weighted MSE
        # where the weights correspond to the diferent p values for prioritization of each sample


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import gym
import time
# With rendering
# env = gym.make('ALE/Pong-v5', render_mode = "human")

# W/o rendering
# env = gym.make('ALE/Pong-v5', render_mode = "rgb_array")
env = gym.make('ALE/Pong-v5')
# observation = env.reset()
# action = env.action_space.sample()
#
# observation, reward, done, info = env.step(action)
# TODO: DQN Agent and Network

net_output_size = env.action_space.n
print("Output size: ", net_output_size)
Agent = DQN_agent(n_actions = net_output_size)

replay_buffer_Full = False
curr_episode = -1
do_random_actions = True
max_episodes = 1000000
# Taking random actions
# By default, as a simplifying measure, demonstrations are collected as random actions using the command
# env.action_space.sample() command


# Importing Image and ImageOps module from PIL package
from PIL import Image, ImageOps

# Making stacks for new and old observations
# image_stack_old = [0 , 0 , 0 , 0]
# image_stack_new = [0 , 0 , 0 , 0]

img_width = 84
img_height = 84

def preprocess(input_array):
    # Convert to grascale and plot
    input_array = np.array(input_array)
    im2 = Image.fromarray((input_array).astype(np.uint8))
    im2 = ImageOps.grayscale(im2)
    # Resize to proper size [84,84]
    im2 = im2.resize([img_width,img_height])
    return np.array(im2)

t=0
total_reward = 0
while do_random_actions:
    image_stack_old = [0, 0, 0, 0]
    image_stack_new = [0, 0, 0, 0]
    episode_not_done = 1
    curr_episode += 1
    observation_old, info = env.reset()
    old_action = 0
    action = 0

    observation_old = preprocess(observation_old)

    print("Episode number:", curr_episode)
    print("Nr Iters:",t)
    print("Reward:",total_reward)
    # env.render()
    curr_episode_length = 0
    t = 0
    total_reward = 0

    # if curr_episode > 0 and curr_episode % 20 == 0:
    #     Agent.save_checkpoint()
    while episode_not_done:
        # env.render()
        # print(observation)
        # action = env.action_space.sample()

        # observation_fixed_axis = np.moveaxis(observation_old,  -1, 0)

        # Restack and push images
        image_stack_old[3] = image_stack_old[2]
        image_stack_old[2] = image_stack_old[1]
        image_stack_old[1] = image_stack_old[0]
        image_stack_old[0] = preprocess(observation_old)

        # observation_old_torch = observation_old_torch.unsqueeze(dim=0)

        # Select a new action every 6 frames
        if t % 6 == 0 and t > 0:
            observation_old_torch = torch.tensor(np.array(image_stack_old)).cuda().float()
            observation_old_torch = observation_old_torch.unsqueeze(dim=0)
            # print(np.shape(observation_old_torch))
            # print(t)
            action = Agent.act(observation_old_torch)
            # print(action)
        # print(action)
        observation_new, reward, done, info, _ = env.step(action)

        processed_observation_new = preprocess(observation_new)

        image_stack_new[0] = preprocess(observation_new)
        image_stack_new[1] = image_stack_old[0]
        image_stack_new[2] = image_stack_old[1]
        image_stack_new[3] = image_stack_old[2]
        # Plotting RGB image
        # plt.imshow(observation_old)
        # plt.show()

        # Convert to grascale and plot
        # im2 = Image.fromarray((observation_old).astype(np.uint8))
        # im2 = ImageOps.grayscale(im2)
        # Resize to proper size [84,84]
        # im2 = im2.resize([img_width,img_height])
        # print(np.shape(im2))
        # # print(np.shape(im2))
        #
        # im2.show()
        # import time
        # time.sleep(5)


        # Dont forget to sleep if env.render = on]
        # time.sleep(2)
        # plt.imshow(observation_old/255)
        # plt.show(block = False)

        if t > 0 and t % 6 == 0:
            transition = [np.array(image_stack_old) , action, float(reward), np.array(image_stack_new), done, False, 0.1, \
                      t -1 , curr_episode]
            Agent.memory.store_sample(transition)
            Agent.update()
        old_action = action
        # Convert previous observation into Image (s)
        # gray_old = cv2.cvtColor(observation_old, cv2.COLOR_BGR2GRAY)
        # resized_image_old = cv2.resize(gray_old, (84, 84))
        #
        # # Convert new state into Image (s_)
        # gray_new = cv2.cvtColor(observation_new, cv2.COLOR_BGR2GRAY)
        # resized_image_new = cv2.resize(gray_new, (84, 84))
        #
        # # Transitions are stored as S, A, R, S_, done, is_demo, weight, idx of sample, curr_episode
        # transition = [resized_image_old, action, reward, resized_image_new, done, True, demo_sample_weight\
        #               , t, curr_episode]
        # transition = np.array(transition)

        # print(action)
        # If the replay buffer is full then stop collecting demonstrations
        # if done:
        #     imgplot = plt.imshow(resized_image_new)
        #     plt.show()
        #     time.sleep(2)
        # Checks if episode has been concluded

        if done:
            episode_not_done = 0

        if curr_episode >= max_episodes:
            do_random_actions = False
        total_reward += reward

        from copy import deepcopy as dp
        observation_old = dp(observation_new)
        t += 1
        curr_episode_length = dp(t)

    Agent.rewards.append(total_reward)
    Agent.iters_per_episode.append(t)
env.close()
