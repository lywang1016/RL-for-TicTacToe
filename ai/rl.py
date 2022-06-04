import yaml
import torch
import random
from os.path import exists
import copy
import time
import numpy as np
from collections import namedtuple, deque
from framework.board import ChessBoard
from framework.display import GUI
from framework.player import HumanPlayer, AIPlayer
from ai.network import DQN
from ai.loss import MyLoss
from ai.reward import reward_function

# check if use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load configuration
with open('ai/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# define dataset
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# game objects
memory = ReplayMemory(10000)

policy_net = DQN().to(device)  # Q*(s,a)
target_net = DQN().to(device)
if exists(config['save_model_path']):
    checkpoint = torch.load(config['save_model_path'])
    policy_net.load_state_dict(checkpoint['model_state_dict'])
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
policy_net.train()

optimizer = torch.optim.RMSprop(policy_net.parameters())

def optimize_model():
    global policy_net

    if len(memory) < config['batch_size']:
        return 2

    transitions = memory.sample(config['batch_size'])
    batch = Transition(*zip(*transitions))

    state = torch.from_numpy(batch.state[0]).to(device)
    action = torch.from_numpy(batch.action[0]).to(device)
    reward = torch.tensor(batch.reward[0]).to(device)
    state_batch = state.view(1, 1, 3, 3)
    action_batch = action.view(1, 1, 3, 3)
    reward_batch = reward.view(1, 1)
    for i in range(1, config['batch_size']):
        state = torch.from_numpy(batch.state[i]).to(device)
        action = torch.from_numpy(batch.action[i]).to(device)
        reward = torch.tensor(batch.reward[i]).to(device)
        state_batch = torch.cat((state_batch, state.view(1, 1, 3, 3)), dim=0)
        action_batch = torch.cat((action_batch, action.view(1, 1, 3, 3)), dim=0)
        reward_batch = torch.cat((reward_batch, reward.view(1, 1)), dim=0)

    next_values = torch.zeros((config['batch_size'],1), device=device)
    for i in range(config['batch_size']):
        next_state = batch.next_state[i]
        actions = []
        for i in range(3):
            for j in range(3):
                if next_state[i][j] == 0:
                    temp = copy.deepcopy(next_state)
                    temp[i][j] = 1
                    actions.append(temp)
        next_state = torch.from_numpy(next_state).to(device)
        next_state = next_state.view(1, 1, 3, 3)
        val_list = []
        for action in actions:
            a = torch.from_numpy(action).to(device)
            a = a.float().view(1, 1, 3, 3)
            value = target_net(next_state, a)
            value = value.cpu().detach().numpy()[0][0]
            val_list.append(value)
        max_val = max(val_list)
        next_values[i][0] = torch.tensor(max_val).to(device)
    next_values = (next_values * config['gamma']) + reward_batch

    cur_values = policy_net(state_batch, action_batch)

    criterion = MyLoss().to(device)
    loss = criterion(cur_values, next_values)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    return float(loss.cpu().detach())






chess_board = ChessBoard()
gui = GUI()


r_player = AIPlayer('r', config['eps_start'])
b_player = AIPlayer('b', config['eps_start'])

