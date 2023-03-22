import os
import yaml
import h5py
import copy
import numpy as np
from os.path import exists
from utils import board_to_list, board_rotate_lr, board_rotate_180, board_rotate_180lr
from constant import posi_idx_map, idx_rotate_180, idx_rotate_lr, idx_rotate_180lr
from player import Player

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
        cur_state_origin = board_to_list(self.current_board)
        cur_state_180 = board_to_list(board_rotate_180(self.current_board))
        cur_state_lr = board_to_list(board_rotate_lr(self.current_board))
        cur_state_180lr = board_to_list(board_rotate_180lr(self.current_board))
        best_idx = -1
        best_q = -1000
        for i in range(length):
            idx = posi_idx_map[self.all_move[i]] 
            temp_a_origin = self.all_move[i]
            temp_a_180 = idx_rotate_180[idx]
            temp_a_lr = idx_rotate_lr[idx]
            temp_a_180lr = idx_rotate_180lr[idx]
            temp_sa_origin = tuple(cur_state_origin + [temp_a_origin[0], temp_a_origin[1]])
            temp_sa_180 = tuple(cur_state_180 + [temp_a_180[0], temp_a_180[1]])
            temp_sa_lr = tuple(cur_state_lr + [temp_a_lr[0], temp_a_lr[1]])
            temp_sa_180lr = tuple(cur_state_180lr + [temp_a_180lr[0], temp_a_180lr[1]])
            if temp_sa_origin in self.q:
                if self.q[temp_sa_origin] > best_q:
                    best_q = self.q[temp_sa_origin]
                    best_idx = i
            elif temp_sa_180 in self.q:
                if self.q[temp_sa_180] > best_q:
                    best_q = self.q[temp_sa_180]
                    best_idx = i
            elif temp_sa_lr in self.q:
                if self.q[temp_sa_lr] > best_q:
                    best_q = self.q[temp_sa_lr]
                    best_idx = i
            elif temp_sa_180lr in self.q:
                if self.q[temp_sa_180lr] > best_q:
                    best_q = self.q[temp_sa_180lr]
                    best_idx = i
            else: 
                self.q[temp_sa_origin] = np.random.rand()
                self.sa_touched.append(temp_sa_origin)
                if self.q[temp_sa_origin] > best_q:
                    best_q = self.q[temp_sa_origin]
                    best_idx = i
        self.action_take = self.all_move[best_idx]
        return self.action_take, self.faction

    def random_action(self):
        length = self.check_moves()
        posi_idx = np.random.randint(length)
        self.action_take = self.all_move[posi_idx]
        return self.action_take, self.faction
    
    def q_update(self, board, whos, termination, win):
        s_prime_origin = board_to_list(board)
        s_prime_180 = board_to_list(board_rotate_180(board))
        s_prime_lr = board_to_list(board_rotate_lr(board))
        s_prime_180lr = board_to_list(board_rotate_180lr(board))
        if whos == 'r':     # red just made an action
            if termination: 
                new_sa_origin = tuple(s_prime_origin + [-1, -1])
                new_sa_180 = tuple(s_prime_180 + [-1, -1])
                new_sa_lr = tuple(s_prime_lr + [-1, -1])
                new_sa_180lr = tuple(s_prime_180lr + [-1, -1])
                if new_sa_origin in self.q:
                    new_sa = new_sa_origin
                elif new_sa_180 in self.q:
                    new_sa = new_sa_180
                elif new_sa_lr in self.q:
                    new_sa = new_sa_lr
                elif new_sa_180lr in self.q:
                    new_sa = new_sa_180lr
                else:
                    self.q[new_sa_origin] = 0
                    self.sa_touched.append(new_sa_origin)  
                    new_sa = new_sa_origin         
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
                    idx = posi_idx_map[all_move[i]]
                    temp_a_origin = all_move[i]
                    temp_a_180 = idx_rotate_180[idx]
                    temp_a_lr = idx_rotate_lr[idx]
                    temp_a_180lr = idx_rotate_180lr[idx]
                    temp_sa_origin = tuple(s_prime_origin + [temp_a_origin[0], temp_a_origin[1]])
                    temp_sa_180 = tuple(s_prime_180 + [temp_a_180[0], temp_a_180[1]])
                    temp_sa_lr = tuple(s_prime_lr + [temp_a_lr[0], temp_a_lr[1]])
                    temp_sa_180lr = tuple(s_prime_180lr + [temp_a_180lr[0], temp_a_180lr[1]])
                    if temp_sa_origin in self.q:
                        if self.q[temp_sa_origin] > best_q:
                            best_q = self.q[temp_sa_origin]
                            best_idx = i
                    elif temp_sa_180 in self.q:
                        if self.q[temp_sa_180] > best_q:
                            best_q = self.q[temp_sa_180]
                            best_idx = i
                    elif temp_sa_lr in self.q:
                        if self.q[temp_sa_lr] > best_q:
                            best_q = self.q[temp_sa_lr]
                            best_idx = i
                    elif temp_sa_180lr in self.q:
                        if self.q[temp_sa_180lr] > best_q:
                            best_q = self.q[temp_sa_180lr]
                            best_idx = i
                    else: 
                        self.q[temp_sa_origin] = np.random.rand()
                        self.sa_touched.append(temp_sa_origin)
                        if self.q[temp_sa_origin] > best_q:
                            best_q = self.q[temp_sa_origin]
                            best_idx = i      
                best_action = all_move[best_idx]
                idx = posi_idx_map[best_action]
                a_origin = best_action
                a_180 = idx_rotate_180[idx]
                a_lr = idx_rotate_lr[idx]
                a_180lr = idx_rotate_180lr[idx]
                new_sa_origin = tuple(s_prime_origin + [a_origin[0], a_origin[1]])
                new_sa_180 = tuple(s_prime_180 + [a_180[0], a_180[1]])
                new_sa_lr = tuple(s_prime_lr + [a_lr[0], a_lr[1]])
                new_sa_180lr = tuple(s_prime_180lr + [a_180lr[0], a_180lr[1]])
                if new_sa_origin in self.q:
                    new_sa = new_sa_origin
                elif new_sa_180 in self.q:
                    new_sa = new_sa_180
                elif new_sa_lr in self.q:
                    new_sa = new_sa_lr
                else:
                    new_sa = new_sa_180lr
                reward = 0
            s_origin = board_to_list(self.current_board)
            s_180 = board_to_list(board_rotate_180(self.current_board))
            s_lr = board_to_list(board_rotate_lr(self.current_board))
            s_180lr = board_to_list(board_rotate_180lr(self.current_board))
            idx = posi_idx_map[self.action_take] 
            a_origin = self.action_take
            a_180 = idx_rotate_180[idx]
            a_lr = idx_rotate_lr[idx]
            a_180lr = idx_rotate_180lr[idx]
            sa_origin = tuple(s_origin + [a_origin[0], a_origin[1]])
            sa_180 = tuple(s_180 + [a_180[0], a_180[1]])
            sa_lr = tuple(s_lr + [a_lr[0], a_lr[1]])
            sa_180lr = tuple(s_180lr + [a_180lr[0], a_180lr[1]])
            if sa_origin in self.q:
                self.sa_touched.append(sa_origin)
                self.q[sa_origin] = self.q[sa_origin] + self.lr*(reward + self.gamma*self.q[new_sa] - self.q[sa_origin])
            elif sa_180 in self.q:
                self.sa_touched.append(sa_180)
                self.q[sa_180] = self.q[sa_180] + self.lr*(reward + self.gamma*self.q[new_sa] - self.q[sa_180])
            elif sa_lr in self.q:
                self.sa_touched.append(sa_lr)
                self.q[sa_lr] = self.q[sa_lr] + self.lr*(reward + self.gamma*self.q[new_sa] - self.q[sa_lr])
            elif sa_180lr in self.q:
                self.sa_touched.append(sa_180lr)
                self.q[sa_180lr] = self.q[sa_180lr] + self.lr*(reward + self.gamma*self.q[new_sa] - self.q[sa_180lr])
            else: 
                self.q[sa_origin] = np.random.rand()  
                self.sa_touched.append(sa_origin)
                self.q[sa_origin] = self.q[sa_origin] + self.lr*(reward + self.gamma*self.q[new_sa] - self.q[sa_origin])
        else:   # black just made an action
            if termination:
                new_sa_origin = tuple(s_prime_origin + [-1, -1])
                new_sa_180 = tuple(s_prime_180 + [-1, -1])
                new_sa_lr = tuple(s_prime_lr + [-1, -1])
                new_sa_180lr = tuple(s_prime_180lr + [-1, -1])
                if new_sa_origin in self.q:
                    new_sa = new_sa_origin
                elif new_sa_180 in self.q:
                    new_sa = new_sa_180
                elif new_sa_lr in self.q:
                    new_sa = new_sa_lr
                elif new_sa_180lr in self.q:
                    new_sa = new_sa_180lr
                else:
                    self.q[new_sa_origin] = 0
                    self.sa_touched.append(new_sa_origin)  
                    new_sa = new_sa_origin      
                if win == 'r':
                    reward = -1
                elif win == 'b':
                    reward = 1
                else:
                    reward = 0
                s_origin = board_to_list(self.current_board)
                s_180 = board_to_list(board_rotate_180(self.current_board))
                s_lr = board_to_list(board_rotate_lr(self.current_board))
                s_180lr = board_to_list(board_rotate_180lr(self.current_board))
                idx = posi_idx_map[self.action_take] 
                a_origin = self.action_take
                a_180 = idx_rotate_180[idx]
                a_lr = idx_rotate_lr[idx]
                a_180lr = idx_rotate_180lr[idx]
                sa_origin = tuple(s_origin + [a_origin[0], a_origin[1]])
                sa_180 = tuple(s_180 + [a_180[0], a_180[1]])
                sa_lr = tuple(s_lr + [a_lr[0], a_lr[1]])
                sa_180lr = tuple(s_180lr + [a_180lr[0], a_180lr[1]])
                if sa_origin in self.q:
                    self.sa_touched.append(sa_origin)
                    self.q[sa_origin] = self.q[sa_origin] + self.lr*(reward + self.gamma*self.q[new_sa] - self.q[sa_origin])
                elif sa_180 in self.q:
                    self.sa_touched.append(sa_180)
                    self.q[sa_180] = self.q[sa_180] + self.lr*(reward + self.gamma*self.q[new_sa] - self.q[sa_180])
                elif sa_lr in self.q:
                    self.sa_touched.append(sa_lr)
                    self.q[sa_lr] = self.q[sa_lr] + self.lr*(reward + self.gamma*self.q[new_sa] - self.q[sa_lr])
                elif sa_180lr in self.q:
                    self.sa_touched.append(sa_180lr)
                    self.q[sa_180lr] = self.q[sa_180lr] + self.lr*(reward + self.gamma*self.q[new_sa] - self.q[sa_180lr])
                else: 
                    self.q[sa_origin] = np.random.rand()  
                    self.sa_touched.append(sa_origin)
                    self.q[sa_origin] = self.q[sa_origin] + self.lr*(reward + self.gamma*self.q[new_sa] - self.q[sa_origin])
