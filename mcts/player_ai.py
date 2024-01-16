import copy
import random
import numpy as np
from tictactoe import TicTacToe
from mcts import MCTS

class AIPlayer:
    def __init__(self, color):
        self.color = color
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
        self.valid_moves = self.tictactoe.get_valid_moves(state)
        return len(self.valid_moves)

    def random_action(self):
        length = self.check_moves()
        if length > 0:
            action = random.choice(self.valid_moves)
            position = self.tictactoe.actions_decode[action]
            if self.color != 'r':
                return position, -1
            else:
                return position, 1
        else:
            return None
        
    def mcts_action(self):
        length = self.check_moves()
        if length > 0:
            if self.color != 'r':
                state = self.tictactoe.change_perspective(copy.deepcopy(self.state), -1)
            else:
                state = copy.deepcopy(self.state)
            mcts_probs = self.mcts.search(state)
            action = np.argmax(mcts_probs)
            position = self.tictactoe.actions_decode[action]
            if self.color != 'r':
                return position, -1
            else:
                return position, 1
        else:
            return None