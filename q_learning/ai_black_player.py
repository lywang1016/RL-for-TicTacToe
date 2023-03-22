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
        self.action_take = None
        self.all_move = []
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
        self.write_h5()
        self.current_board = None
        self.action_take = None
        self.all_move = []
        self.sa_touched = []
    
    def load_h5(self):
        if exists(self.q_path):
            fq = h5py.File(self.q_path, 'r')
            for key in fq:
                q_value = np.array(fq[key])
                temp = key[1:len(key)-1].split(',')
                sa = []
                for item in temp:
                    sa.append(float(item))
                sa = tuple(sa)
                self.q[sa] = q_value
        else:
            fq = h5py.File(self.q_path, "w")
        fq.close()
    
    def write_h5(self):
        self.sa_touched = list(set(self.sa_touched))
        fq = h5py.File(self.q_path, "a")
        for item in self.sa_touched:
            h5_key = str(item)
            if h5_key in fq:
                del fq[h5_key]
            fq.create_dataset(h5_key, data=self.q[item])
        fq.close()
    
    def eps_greedy_action(self):
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
                self.sa_touched.append(temp_sa)
            if self.q[temp_sa] > best_q:
                best_q = self.q[temp_sa]
                best_idx = i
        self.action_take = self.all_move[best_idx]
        return self.action_take, self.faction

    def random_action(self):
        length = self.check_moves()
        posi_idx = np.random.randint(length)
        self.action_take = self.all_move[posi_idx]
        return self.action_take, self.faction
    
    def q_update(self, board, whos, termination, win):
        s_prime = board_to_list(board)
        if whos == 'r':     # red just made an action
            if termination: 
                new_sa = tuple(s_prime + [-1, -1])
                if new_sa not in self.q:
                    self.q[new_sa] = 0
                    self.sa_touched.append(new_sa)
                if win == 'r':
                    reward = -1
                elif win == 'b':
                    reward = 1
                else:
                    reward = 0
            else:
                all_move = []
                for i in range(3):
                    for j in range(3):
                        if board[i][j] == 0:
                            all_move.append((i,j))
                length = len(all_move)
                best_idx = -1
                best_q = -1000
                for i in range(length):
                    temp_sa = tuple(s_prime + [all_move[i][0], all_move[i][1]])
                    if temp_sa not in self.q:
                        self.q[temp_sa] = np.random.rand()
                        self.sa_touched.append(temp_sa)
                    if self.q[temp_sa] > best_q:
                        best_q = self.q[temp_sa]
                        best_idx = i
                best_action = all_move[best_idx]
                new_sa = tuple(s_prime + [best_action[0], best_action[1]])
                reward = 0
            s = board_to_list(self.current_board)
            sa = tuple(s + [self.action_take[0], self.action_take[1]])
            if sa not in self.q:
                self.q[sa] = np.random.rand()
            self.sa_touched.append(sa)
            self.q[sa] = self.q[sa] + self.lr*(reward + self.gamma*self.q[new_sa] - self.q[sa]) 
        else:   # black just made an action
            if termination:
                new_sa = tuple(s_prime + [-1, -1])
                if new_sa not in self.q:
                    self.q[new_sa] = 0
                    self.sa_touched.append(new_sa)
                if win == 'r':
                    reward = -1
                elif win == 'b':
                    reward = 1
                else:
                    reward = 0
                s = board_to_list(self.current_board)
                sa = tuple(s + [self.action_take[0], self.action_take[1]])
                if sa not in self.q:
                    self.q[sa] = np.random.rand()
                self.sa_touched.append(sa)
                self.q[sa] = self.q[sa] + self.lr*(reward + self.gamma*self.q[new_sa] - self.q[sa]) 
