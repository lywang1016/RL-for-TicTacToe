import copy
import random
import numpy as np
import torch as T
from tictactoe import TicTacToe
from mcts import MCTS
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

        args = {
            'C': 2,
            'num_searches': 1000
        }
        self.mcts = MCTS(self.tictactoe, args)

        self.model = ResNet(self.tictactoe, 4, 64)
        self.model.load_state_dict(T.load('model/model.pth'))
        self.model.eval()

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
            # print(mcts_probs)
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
            # print(policy)
            action = np.argmax(policy)
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
    model.load_state_dict(T.load('model/model_2.pth'))
    model.eval()

    state = tictactoe.get_initial_state()
    state[0,2] = 1
    state[2,1] = -1

    mcts_probs = mcts.search(state.copy(), player=1)
    action = np.argmax(mcts_probs)
    print(action)
    plt.bar(range(tictactoe.action_size), mcts_probs)
    plt.show()

    encoded_state = tictactoe.get_encoded_state(state.copy(), player=1)
    tensor_state = T.tensor(encoded_state).float().to(model.device).unsqueeze(0)
    policy, value = model(tensor_state)
    value = value.item()
    policy = T.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
    plt.bar(range(tictactoe.action_size), policy)
    plt.show()

    valid_moves = tictactoe.get_valid_moves(state.copy(), player=1)
    mask = tictactoe.valid_moves_mask(valid_moves)
    policy *= mask
    policy /= np.sum(policy)
    plt.bar(range(tictactoe.action_size), policy)
    plt.show()