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
        self.action_take = None
        self.all_move = []
        self.first_move = True
        self.sa_touched = []
        self.q = {}
        with open('config.yaml') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.eps = self.config['eps']
        self.lr = self.config['learn_rate']
        self.gamma = self.config['discount']
        pwd = os.getcwd()
        root_dir = os.path.abspath(os.path.dirname(pwd) + os.path.sep + '.')
        self.q_path = os.path.join(root_dir, self.config['black_ai_q_val_path'])
        self.load_h5()

    def reset(self):
        self.current_board = None
        self.last_board = None
        self.action_take = None
        self.all_move = []
        self.first_move = True
        self.sa_touched = []
    
    def update_board(self, board):
        self.last_board = self.current_board
        self.current_board = copy.deepcopy(board)
    
    def load_h5(self):
        if exists(self.q_path):
            fq = h5py.File(self.q_path, 'r')
            for key in fq:
                print(key)
        else:
            fq = h5py.File(self.q_path, "w")
        fq.close()
    
    def write_h5(self):
        fq = h5py.File(self.q_path, "a")
        for item in self.sa_touched:
            h5_key = str(item)
            if h5_key in fq:
                del fq[h5_key]
            fq.create_dataset(h5_key, self.q[item])
        fq.close()
    
    def eps_greedy_action(self):
        if self.first_move:
            self.first_move = False
            self.write_h5()
        else:
            self.q_update()
        temp = np.random.rand()
        if temp < self.eps:
            return self.random_action()
        else:
            return self.greedy_action()

    def greedy_action(self):
        length = self.check_moves()
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
        self.action_take = self.all_move[best_idx]
        return self.action_take, self.faction

    def random_action(self):
        posi_num = len(self.all_move)
        posi_idx = np.random.randint(posi_num)
        self.action_take = self.all_move[posi_idx]
        return self.action_take, self.faction
    
    def q_update(self):
        pre_state = board_to_list(self.last_board)
        sa = tuple(pre_state + [self.action_take[0], self.action_take[1]])
        self.sa_touched.append(sa)
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
                        self.sa_touched.append(temp_sa)
                    if self.q[temp_sa] > best_q:
                        best_q = self.q[temp_sa]
                        best_idx = i
                best_action = self.all_move[best_idx]
                new_sa = tuple(cur_state + [best_action[0], best_action[1]])
                if new_sa not in self.q:
                    self.q[new_sa] = np.random.rand()
                    self.sa_touched.append(new_sa)
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
