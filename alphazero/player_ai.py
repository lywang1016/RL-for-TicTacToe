import copy
import random
import yaml
import numpy as np
import torch as T
from tictactoe import TicTacToe
from mcts import MCTS
from alpha_mcts import AlphaMCTS
from network import ResNet
from utils import action_change_perspective

class AIPlayer:
    def __init__(self, color):
        if color == 'r':
            self.player = 1
        else:
            self.player = -1
        self.tictactoe = TicTacToe()
        self.state = None
        self.valid_moves = []

        with open('config.yaml') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        args = {
            'C': self.config['C'],
            'num_searches': self.config['num_MCTS_searches']
        }
        self.mcts = MCTS(self.tictactoe, args)

        self.model = ResNet(self.tictactoe, self.config['num_resBlocks'], self.config['num_hidden'])
        self.model.load_state_dict(T.load(self.config['model_full_path']))
        self.model.eval()

        args = {
            'C': self.config['C'],
            'num_searches': self.config['num_Alpha_MCTS_searches'],
            'dirichlet_epsilon': 0.0,       # for evaluation we don't want add noise
            'dirichlet_alpha': 0.1,
        }
        self.alpha_mcts = AlphaMCTS(self.tictactoe, args, self.model)

    def reset(self):
        self.state = None
        self.valid_moves = []

    def update_board(self, board):
        self.state = copy.deepcopy(board)

    def check_moves(self):
        state = copy.deepcopy(self.state)
        self.valid_moves = self.tictactoe.get_valid_moves(state, self.player)
        return len(self.valid_moves)

    def random_action(self):
        length = self.check_moves()
        if length > 0:
            action = random.choice(self.valid_moves)
            position = self.tictactoe.actions_decode[action]
            if self.player != 1:
                position = action_change_perspective(position)
            return position, self.player
        else:
            return None
        
    def mcts_action(self):
        length = self.check_moves()
        if length > 0:
            state = copy.deepcopy(self.state)
            mcts_probs = self.mcts.search(state, self.player)
            action = np.argmax(mcts_probs)
            position = self.tictactoe.actions_decode[action]
            if self.player != 1:
                position = action_change_perspective(position)
            return position, self.player
        else:
            return None
        
    @T.no_grad()  
    def model_action(self):
        length = self.check_moves()
        if length > 0:
            state = copy.deepcopy(self.state)
            encoded_state = self.tictactoe.get_encoded_state(state, self.player)
            policy, value = self.model(
                T.tensor(encoded_state).float().to(self.model.device).unsqueeze(0)
            )
            policy = T.softmax(policy, axis=1).squeeze(0).cpu().numpy()
            value = value.item()
            valid_moves = self.tictactoe.get_valid_moves(self.state.copy(), self.player)
            mask = self.tictactoe.valid_moves_mask(valid_moves)
            policy *= mask
            policy /= np.sum(policy)
            action = np.argmax(policy)
            position = self.tictactoe.actions_decode[action]
            if self.player != 1:
                position = action_change_perspective(position)
            return position, self.player
        else:
            return None
        
    def alpha_mcts_action(self):
        length = self.check_moves()
        if length > 0:
            state = copy.deepcopy(self.state)
            mcts_probs = self.alpha_mcts.search(state, self.player)
            action = np.argmax(mcts_probs)
            position = self.tictactoe.actions_decode[action]
            if self.player != 1:
                position = action_change_perspective(position)
            return position, self.player
        else:
            return None
        
if __name__ == '__main__':
    # tictactoe = TicTacToe()
    # player = AIPlayer('r')

    # state = tictactoe.get_initial_state()
    # player.update_board(state)
    # if player.check_moves() > 0:
    #     posi, move = player.mcts_action()
    #     print(posi, move)
    #     posi, move = player.model_action()
    #     print(posi, move)

    import matplotlib.pyplot as plt

    tictactoe = TicTacToe()

    args = {
        'C': 2,
        'num_searches': 2000
    }
    mcts = MCTS(tictactoe, args)

    model = ResNet(tictactoe, 4, 64)
    model.load_state_dict(T.load('model/model.pth'))
    model.eval()

    args = {
        'C': 2,
        'num_searches': 20,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3,
    }
    alpha_mcts = AlphaMCTS(tictactoe, args, model)

    state = tictactoe.get_initial_state()
    state[1,1] = 1
    state[2,1] = 1
    state[2,2] = -1
    player = -1

    mcts_probs = mcts.search(state.copy(), player=player)
    action = np.argmax(mcts_probs)
    print(action)
    plt.bar(range(tictactoe.action_size), mcts_probs)
    plt.show()

    encoded_state = tictactoe.get_encoded_state(state.copy(), player=player)
    tensor_state = T.tensor(encoded_state).float().to(model.device).unsqueeze(0)
    policy, value = model(tensor_state)
    value = value.item()
    policy = T.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
    # plt.bar(range(tictactoe.action_size), policy)
    # plt.show()
    valid_moves = tictactoe.get_valid_moves(state.copy(), player=player)
    mask = tictactoe.valid_moves_mask(valid_moves)
    policy *= mask
    policy /= np.sum(policy)
    action = np.argmax(policy)
    print(action)
    plt.bar(range(tictactoe.action_size), policy)
    plt.show()

    mcts_probs = alpha_mcts.search(state.copy(), player=player)
    action = np.argmax(mcts_probs)
    print(action)
    plt.bar(range(tictactoe.action_size), mcts_probs)
    plt.show()