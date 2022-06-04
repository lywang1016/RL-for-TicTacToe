import yaml
import torch
import random
import numpy as np
from collections import namedtuple, deque
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
