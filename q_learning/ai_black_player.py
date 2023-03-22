import os
import yaml
import copy
import h5py
import numpy as np
from os.path import exists
from utils import board_to_list
from player import Player
from board import ChessBoard

class AIBlackPlayer(Player):
    def __init__(self):
        self.faction = -1
        self.current_board = None
        self.last_board = None
        self.last_action = None
        self.all_move = []
        self.q = {}
        with open('config.yaml') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.eps = self.config['eps']
        self.lr = self.config['learn_rate']
        self.gamma = self.config['discount']
        pwd = os.getcwd()
        root_dir = os.path.abspath(os.path.dirname(pwd) + os.path.sep + '..')
        self.q_path = os.path.join(root_dir, self.config['black_ai_q_val_path'])
        if exists(self.q_path):
            fq = h5py.File(self.q_path, 'r')
            for key in fq:
                print(key)

    def reset(self):
        self.current_board = None
        self.last_board = None
        self.last_action = None
        self.all_move = []
    
    def update_board(self, board):
        self.last_board = self.current_board
        self.current_board = copy.deepcopy(board)
    
    def eps_greedy_action(self):
        temp = np.random.rand()
        if temp < self.eps:
            return self.random_action()
        else:
            return self.greedy_action()

    def greedy_action(self):
        return self.random_action()
        # bk = board_to_list(self.current_board)

        # if exists(self.q_path):
        #     fx = h5py.File(self.q_path, 'r')
            
        # else:
        #     return self.random_action()

    def random_action(self):
        state = board_to_list(self.current_board)
        posi_num = len(self.all_move)
        for i in range(posi_num):
            action = self.all_move[i]
            sa = tuple(state + [action[0], action[1]])
            if sa not in self.q:
                self.q[sa] = np.random.rand()
        posi_idx = np.random.randint(posi_num)
        return self.all_move[posi_idx], self.faction
    
    def q_update(self):
        pre_state = board_to_list(self.last_board)
        sa = tuple(pre_state + [self.last_action[0], self.last_action[1]])
        if sa not in self.q:
            self.q[sa] = np.random.rand()
        else:
            length = self.check_moves()
            if length > 1:  # not a terminate state    
                cur_state = board_to_list(self.current_board)
                best_idx = -1
                best_q = -1000
                for i in range(length):
                    temp_sa = tuple(cur_state + [self.all_move[i][0], self.all_move[i][1]])
                    if temp_sa not in self.q:
                        self.q[temp_sa] = np.random.rand()
                    if self.q[temp_sa] > best_q:
                        best_q = self.q[temp_sa]
                        best_idx = i
                best_action = self.all_move[best_idx]
                new_sa = tuple(cur_state + [best_action[0], best_action[1]])
                if new_sa not in self.q:
                    self.q[new_sa] = np.random.rand()
                self.q[sa] = self.q[sa] + self.lr*(self.gamma*self.q[new_sa] - self.q[sa])
            else: # terminate state
                temp_board = ChessBoard()
                temp_board.load_board(self.current_board)
                if length == 1:
                    temp_board.move_piece(self.all_move[0], self.faction)
                if temp_board.win == 'r':   # ai lose
                    reward = -1
                elif temp_board.win == 'b':   # ai win
                    reward = 1
                else:                       # tie
                    reward = 0
                self.q[sa] = self.q[sa] + self.lr*(reward - self.q[sa])
