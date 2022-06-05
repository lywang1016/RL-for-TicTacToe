import yaml
import torch
import random
import copy
import time
import math
import numpy as np
from tqdm import tqdm
from os.path import exists
from collections import namedtuple, deque
from framework.board import ChessBoard
from framework.display import GUI
from framework.player import HumanPlayer, AIPlayer
from framework.utils import board_turn180
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
        memory = deque([],maxlen=capacity)

    def push(self, *args):
        memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(memory, batch_size)

    def __len__(self):
        return len(memory)

# game objects
memory = ReplayMemory(10000)

policy_net = DQN().to(device)  # Q*(s,a)
optimizer = torch.optim.RMSprop(policy_net.parameters())

if exists(config['save_model_path']):
    checkpoint = torch.load(config['save_model_path'])
    policy_net.load_state_dict(checkpoint['model_state_dict'])
else:
    state = {'model_state_dict': policy_net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    torch.save(state, config['save_model_path'])
policy_net.train()

target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

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



loss_history = []
for i_episode in tqdm(range(config['total_episode_num'])):
    chess_board.reset_board()
    r_player.reset()
    b_player.reset()
    explore_rate = config['eps_end'] + (config['eps_start'] - config['eps_end']) * \
        math.exp(-1. * i_episode / config['eps_decay'])
    r_player.set_explore_rate(explore_rate)
    b_player.set_explore_rate(explore_rate)
    total_loss = 0  
    red = True
    b_move = False
    cnt = 0

    while not chess_board.done:
        gui.check_event()
        
        if red:
            gui.update(chess_board.board_states(), 'r')

            r_player.update_board(chess_board.board_states())
            if not r_player.check_moves():
                chess_board.set_done('b')
                break
            posi, move = r_player.ai_action()
            chess_board.move_piece(posi, move)
            red = not red

            r_state = r_player.current_board
            r_action = chess_board.board_states()
            r_reward = reward_function(r_action)

            if b_move:
                b_next_state = board_turn180(chess_board.board_states())
                memory.push(b_state, b_action, b_next_state, b_reward)
                loss = optimize_model()
                total_loss += loss
                cnt += 1
            
        else:
            gui.update(chess_board.board_states(), 'b')

            b_player.update_board(chess_board.board_states())
            if not b_player.check_moves():
                chess_board.set_done('r')
                break
            posi, move = b_player.ai_action()
            chess_board.move_piece(posi, move)
            b_move = True
            red = not red

            b_state = b_player.current_board
            b_action = board_turn180(chess_board.board_states())
            b_reward = reward_function(b_action)

            r_next_state = chess_board.board_states()
            memory.push(r_state, r_action, r_next_state, r_reward)
            loss = optimize_model()
            total_loss += loss
            cnt += 1

    print('Last episode loss is: ' + str(total_loss/cnt))
    loss_history.append(total_loss/cnt)

    if i_episode % config['target_update'] == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if i_episode % config['save_every'] == 0:
        save_path = config['save_model_path']
        state = {'model_state_dict': target_net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
        torch.save(state, save_path)

save_path = config['save_model_path']
state = {'model_state_dict': policy_net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
torch.save(state, save_path)