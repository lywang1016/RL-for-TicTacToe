import copy
import random
import numpy as np
from collections import deque
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
        if self.color != 'r':
            state = self.tictactoe.change_perspective(copy.deepcopy(self.state), -1)
        else:
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
        
    # def mcts_action(self):
    #     length = self.check_moves()
    #     if length > 0:
    #         if self.color != 'r':
    #             state = self.tictactoe.change_perspective(copy.deepcopy(self.state), -1)
    #         else:
    #             state = copy.deepcopy(self.state)
    #         mcts_probs = self.mcts.search('r', state, self.step_cnt)
    #         action = np.argmax(mcts_probs)
    #         decoded_action = self.tictactoe.actions_decode[action]
    #         if (decoded_action[1], decoded_action[2]) == self.banned_posi:
    #             mcts_probs[action] = 0
    #             action = np.argmax(mcts_probs)
    #             decoded_action = self.tictactoe.actions_decode[action] 
    #         position = [decoded_action[1], decoded_action[2]]
    #         move = [decoded_action[3], decoded_action[4]]
    #         self.past_actions.append((position, move))
    #         self.step_cnt += 2
    #         if self.color != 'r':
    #             return action_change_perspective(position, move)
    #         else:
    #             return position, move
    #     else:
    #         return None