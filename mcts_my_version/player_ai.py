import copy
import random
import numpy as np
from tictactoe import TicTacToe
from mcts import MCTS
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