import os
import yaml
import h5py
import numpy as np
from os.path import exists
from utils import board_to_list, board_rotate_lr, board_rotate_180, board_rotate_180lr
from constant import posi_idx_map, idx_rotate_180, idx_rotate_lr, idx_rotate_180lr
from player import Player

class AIPlayer(Player):
    def __init__(self, color):
        pwd = os.getcwd()
        root_dir = os.path.abspath(os.path.dirname(pwd) + os.path.sep + '.')

        self.faction = 1
        if color == 'b':
            self.faction = -1
        self.current_board = np.zeros((3, 3))
        self.action_take = (-1, -1)
        self.all_move = []
        self.sa_touched = []
        self.q = {}
        self.q1 = {}
        self.q2 = {}

        with open('config.yaml') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.eps = self.config['eps_r']
        self.lr = self.config['learn_rate_r']
        self.gamma = self.config['discount_r']
        self.learn_method = self.config['learn_method_r']
        if self.learn_method == 'q_learning':
            self.q_path = os.path.join(root_dir, self.config['red_ai_q_learning_val_path'])
        if self.learn_method == 'double_q_learning':
            self.q1_path = os.path.join(root_dir, self.config['red_ai_q1_val_path'])
            self.q2_path = os.path.join(root_dir, self.config['red_ai_q2_val_path'])
        if self.learn_method == 'sarsa':
            self.q_path = os.path.join(root_dir, self.config['red_ai_q_sarsa_val_path'])
            self.next_action = None
            self.init_action = True
        if color == 'b':
            self.learn_method = self.config['learn_method_b']
            self.eps = self.config['eps_b']
            self.lr = self.config['learn_rate_b']
            self.gamma = self.config['discount_b']
            if self.learn_method == 'q_learning':
                self.q_path = os.path.join(root_dir, self.config['black_ai_q_learning_val_path'])
            if self.learn_method == 'double_q_learning':
                self.q1_path = os.path.join(root_dir, self.config['black_ai_q1_val_path'])
                self.q2_path = os.path.join(root_dir, self.config['black_ai_q2_val_path'])
            if self.learn_method == 'sarsa':
                self.q_path = os.path.join(root_dir, self.config['black_ai_q_sarsa_val_path'])
                self.next_action = None
                self.init_action = True
        self.load_h5()

    def reset(self):
        self.write_h5()
        self.current_board = np.zeros((3, 3))
        self.action_take = (-1, -1)
        self.all_move = []
        self.sa_touched = []
        if self.learn_method == 'sarsa':
            self.init_action = True
            self.next_action = None
    
    def load_h5(self):
        if self.learn_method == 'sarsa' or self.learn_method == 'q_learning':
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
        else:
            if exists(self.q1_path):
                fq1 = h5py.File(self.q1_path, 'r')
                for key in fq1:
                    q_value = np.array(fq1[key])
                    temp = key[1:len(key)-1].split(',')
                    sa = []
                    for item in temp:
                        sa.append(float(item))
                    sa = tuple(sa)
                    self.q1[sa] = q_value
            else:
                fq1 = h5py.File(self.q1_path, "w")
            fq1.close()
            if exists(self.q2_path):
                fq2 = h5py.File(self.q2_path, 'r')
                for key in fq2:
                    q_value = np.array(fq2[key])
                    temp = key[1:len(key)-1].split(',')
                    sa = []
                    for item in temp:
                        sa.append(float(item))
                    sa = tuple(sa)
                    self.q2[sa] = q_value
            else:
                fq2 = h5py.File(self.q2_path, "w")
            fq2.close()
    
    def write_h5(self):
        self.sa_touched = list(set(self.sa_touched))
        if self.learn_method == 'sarsa' or self.learn_method == 'q_learning':
            fq = h5py.File(self.q_path, "a")
            for item in self.sa_touched:
                h5_key = str(item)
                if h5_key in fq:
                    del fq[h5_key]
                fq.create_dataset(h5_key, data=self.q[item])
            fq.close()
        else:
            fq1 = h5py.File(self.q1_path, "a")
            for item in self.sa_touched:
                h5_key = str(item)
                if h5_key in fq1:
                    del fq1[h5_key]
                fq1.create_dataset(h5_key, data=self.q1[item])
            fq1.close()
            fq2 = h5py.File(self.q2_path, "a")
            for item in self.sa_touched:
                h5_key = str(item)
                if h5_key in fq2:
                    del fq2[h5_key]
                fq2.create_dataset(h5_key, data=self.q2[item])
            fq2.close()
    
    def eps_greedy_action(self):
        if self.learn_method == 'sarsa':
            if self.init_action:
                self.init_action = False
                temp = np.random.rand()
                if temp < self.eps:
                    return self.random_action()
                else:
                    return self.greedy_action()
            else:
                self.action_take = self.next_action
                return self.action_take, self.faction
        else:
            temp = np.random.rand()
            if temp < self.eps:
                return self.random_action()
            else:
                return self.greedy_action()

    def greedy_action(self):
        if self.learn_method == 'sarsa' or self.learn_method == 'q_learning':
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
        else: 
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
                if temp_sa_origin in self.q1 and temp_sa_origin in self.q2:
                    if self.q1[temp_sa_origin] + self.q2[temp_sa_origin] > best_q:
                        best_q = self.q1[temp_sa_origin] + self.q2[temp_sa_origin]
                        best_idx = i
                elif temp_sa_180 in self.q1 and temp_sa_180 in self.q2:
                    if self.q1[temp_sa_180] + self.q2[temp_sa_180] > best_q:
                        best_q = self.q1[temp_sa_180] + self.q2[temp_sa_180]
                        best_idx = i
                elif temp_sa_lr in self.q1 and temp_sa_lr in self.q2:
                    if self.q1[temp_sa_lr] + self.q2[temp_sa_lr] > best_q:
                        best_q = self.q1[temp_sa_lr] + self.q2[temp_sa_lr]
                        best_idx = i
                elif temp_sa_180lr in self.q1 and temp_sa_180lr in self.q2:
                    if self.q1[temp_sa_180lr] + self.q2[temp_sa_180lr] > best_q:
                        best_q = self.q1[temp_sa_180lr] + self.q2[temp_sa_180lr]
                        best_idx = i
                else: 
                    self.q1[temp_sa_origin] = np.random.rand()
                    self.q2[temp_sa_origin] = np.random.rand()
                    self.sa_touched.append(temp_sa_origin)
                    if self.q1[temp_sa_origin] + self.q2[temp_sa_origin] > best_q:
                        best_q = self.q1[temp_sa_origin] + self.q2[temp_sa_origin]
                        best_idx = i
            self.action_take = self.all_move[best_idx]
            return self.action_take, self.faction

    def random_action(self):
        length = self.check_moves()
        posi_idx = np.random.randint(length)
        self.action_take = self.all_move[posi_idx]
        return self.action_take, self.faction
    
    def q_update(self, board, red_move, termination, win):
        if self.learn_method == 'sarsa':
            self.sarsa_q_learning(board, red_move, termination, win)
        if self.learn_method == 'q_learning':
            self.q_learning(board, red_move, termination, win)
        if self.learn_method == 'double_q_learning':
            self.double_q_learning(board, red_move, termination, win)
    
    def sarsa_q_learning(self, board, red_move, termination, win):
        if self.faction == -1:
            self.__q_sarsa_update_b(board, red_move, termination, win)
        else:
            self.__q_sarsa_update_r(board, red_move, termination, win)
    
    def q_learning(self, board, red_move, termination, win):
        if self.faction == -1:
            self.__q_learning_update_b(board, red_move, termination, win)
        else:
            self.__q_learning_update_r(board, red_move, termination, win)
    
    def double_q_learning(self, board, red_move, termination, win):
        if self.faction == -1:
            self.__double_q_update_b(board, red_move, termination, win)
        else:
            self.__double_q_update_r(board, red_move, termination, win)

    def __q_sarsa_update_r(self, board, red_move, termination, win):
        s_prime_origin = board_to_list(board)
        s_prime_180 = board_to_list(board_rotate_180(board))
        s_prime_lr = board_to_list(board_rotate_lr(board))
        s_prime_180lr = board_to_list(board_rotate_180lr(board))
        if not red_move:     # black just made an action
            if termination:        
                if win == 'r':
                    reward = 2
                elif win == 'b':
                    reward = -100
                else:
                    reward = 0

                new_sa_origin = tuple(s_prime_origin + [-1.0, -1.0])
                new_sa_180 = tuple(s_prime_180 + [-1.0, -1.0])
                new_sa_lr = tuple(s_prime_lr + [-1.0, -1.0])
                new_sa_180lr = tuple(s_prime_180lr + [-1.0, -1.0])
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
            else:          
                reward = 0

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
                temp = np.random.rand()
                if temp < self.eps:
                    posi_idx = np.random.randint(length)
                    self.next_action = all_move[posi_idx]  
                else:
                    self.next_action = best_action  
                idx = posi_idx_map[self.next_action]
                a_origin = self.next_action
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
        else:   # red just made an action
            if termination:
                if win == 'r':
                    reward = 2
                elif win == 'b':
                    reward = -100
                else:
                    reward = 0

                new_sa_origin = tuple(s_prime_origin + [-1.0, -1.0])
                new_sa_180 = tuple(s_prime_180 + [-1.0, -1.0])
                new_sa_lr = tuple(s_prime_lr + [-1.0, -1.0])
                new_sa_180lr = tuple(s_prime_180lr + [-1.0, -1.0])
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

    def __q_sarsa_update_b(self, board, red_move, termination, win):
        s_prime_origin = board_to_list(board)
        s_prime_180 = board_to_list(board_rotate_180(board))
        s_prime_lr = board_to_list(board_rotate_lr(board))
        s_prime_180lr = board_to_list(board_rotate_180lr(board))
        if red_move:     # red just made an action
            if termination:        
                if win == 'r':
                    reward = -100
                elif win == 'b':
                    reward = 2
                else:
                    reward = 0

                new_sa_origin = tuple(s_prime_origin + [-1.0, -1.0])
                new_sa_180 = tuple(s_prime_180 + [-1.0, -1.0])
                new_sa_lr = tuple(s_prime_lr + [-1.0, -1.0])
                new_sa_180lr = tuple(s_prime_180lr + [-1.0, -1.0])
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
            else:           
                reward = 0

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
                temp = np.random.rand()
                if temp < self.eps:
                    posi_idx = np.random.randint(length)
                    self.next_action = all_move[posi_idx]
                else:
                    self.next_action = best_action
                idx = posi_idx_map[self.next_action]
                a_origin = self.next_action
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
                if win == 'r':
                    reward = -100
                elif win == 'b':
                    reward = 2
                else:
                    reward = 0

                new_sa_origin = tuple(s_prime_origin + [-1.0, -1.0])
                new_sa_180 = tuple(s_prime_180 + [-1.0, -1.0])
                new_sa_lr = tuple(s_prime_lr + [-1.0, -1.0])
                new_sa_180lr = tuple(s_prime_180lr + [-1.0, -1.0])
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
    
    def __q_learning_update_r(self, board, red_move, termination, win):
        s_prime_origin = board_to_list(board)
        s_prime_180 = board_to_list(board_rotate_180(board))
        s_prime_lr = board_to_list(board_rotate_lr(board))
        s_prime_180lr = board_to_list(board_rotate_180lr(board))
        if not red_move:     # black just made an action
            if termination:        
                if win == 'r':
                    reward = 2
                elif win == 'b':
                    reward = -100
                else:
                    reward = 0

                new_sa_origin = tuple(s_prime_origin + [-1.0, -1.0])
                new_sa_180 = tuple(s_prime_180 + [-1.0, -1.0])
                new_sa_lr = tuple(s_prime_lr + [-1.0, -1.0])
                new_sa_180lr = tuple(s_prime_180lr + [-1.0, -1.0])
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
            else:           
                reward = 0

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
        else:   # red just made an action
            if termination:
                if win == 'r':
                    reward = 2
                elif win == 'b':
                    reward = -100
                else:
                    reward = 0

                new_sa_origin = tuple(s_prime_origin + [-1.0, -1.0])
                new_sa_180 = tuple(s_prime_180 + [-1.0, -1.0])
                new_sa_lr = tuple(s_prime_lr + [-1.0, -1.0])
                new_sa_180lr = tuple(s_prime_180lr + [-1.0, -1.0])
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

    def __q_learning_update_b(self, board, red_move, termination, win):
        s_prime_origin = board_to_list(board)
        s_prime_180 = board_to_list(board_rotate_180(board))
        s_prime_lr = board_to_list(board_rotate_lr(board))
        s_prime_180lr = board_to_list(board_rotate_180lr(board))
        if red_move:     # red just made an action
            if termination:        
                if win == 'r':
                    reward = -100
                elif win == 'b':
                    reward = 2
                else:
                    reward = 0

                new_sa_origin = tuple(s_prime_origin + [-1.0, -1.0])
                new_sa_180 = tuple(s_prime_180 + [-1.0, -1.0])
                new_sa_lr = tuple(s_prime_lr + [-1.0, -1.0])
                new_sa_180lr = tuple(s_prime_180lr + [-1.0, -1.0])
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
            else:           
                reward = 0

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
                if win == 'r':
                    reward = -100
                elif win == 'b':
                    reward = 2
                else:
                    reward = 0

                new_sa_origin = tuple(s_prime_origin + [-1.0, -1.0])
                new_sa_180 = tuple(s_prime_180 + [-1.0, -1.0])
                new_sa_lr = tuple(s_prime_lr + [-1.0, -1.0])
                new_sa_180lr = tuple(s_prime_180lr + [-1.0, -1.0])
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

    def __double_q_update_r(self, board, red_move, termination, win):
        s_prime_origin = board_to_list(board)
        s_prime_180 = board_to_list(board_rotate_180(board))
        s_prime_lr = board_to_list(board_rotate_lr(board))
        s_prime_180lr = board_to_list(board_rotate_180lr(board))
        if not red_move:     # black just made an action
            if termination: 
                if win == 'r':
                    reward = 1
                elif win == 'b':
                    reward = -1
                else:
                    reward = 0

                new_sa_origin = tuple(s_prime_origin + [-1.0, -1.0])
                new_sa_180 = tuple(s_prime_180 + [-1.0, -1.0])
                new_sa_lr = tuple(s_prime_lr + [-1.0, -1.0])
                new_sa_180lr = tuple(s_prime_180lr + [-1.0, -1.0])
                if new_sa_origin in self.q1 and new_sa_origin in self.q2:
                    new_sa = new_sa_origin
                elif new_sa_180 in self.q1 and new_sa_180 in self.q2:
                    new_sa = new_sa_180
                elif new_sa_lr in self.q1 and new_sa_lr in self.q2:
                    new_sa = new_sa_lr
                elif new_sa_180lr in self.q1 and new_sa_180lr in self.q2:
                    new_sa = new_sa_180lr
                else:
                    self.q1[new_sa_origin] = 0
                    self.q2[new_sa_origin] = 0
                    self.sa_touched.append(new_sa_origin)  
                    new_sa = new_sa_origin 

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

                if np.random.rand() < 0.5:  # update Q1
                    if sa_origin in self.q1 and sa_origin in self.q2:
                        self.sa_touched.append(sa_origin)
                        self.q1[sa_origin] = self.q1[sa_origin] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_origin])
                    elif sa_180 in self.q1 and sa_180 in self.q2:
                        self.sa_touched.append(sa_180)
                        self.q1[sa_180] = self.q1[sa_180] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_180])
                    elif sa_lr in self.q1 and sa_lr in self.q2:
                        self.sa_touched.append(sa_lr)
                        self.q1[sa_lr] = self.q1[sa_lr] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_lr])
                    elif sa_180lr in self.q1 and sa_180lr in self.q2:
                        self.sa_touched.append(sa_180lr)
                        self.q1[sa_180lr] = self.q1[sa_180lr] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_180lr])
                    else: 
                        self.q1[sa_origin] = np.random.rand()  
                        self.q2[sa_origin] = np.random.rand()  
                        self.sa_touched.append(sa_origin)
                        self.q1[sa_origin] = self.q1[sa_origin] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_origin])
                else:   # update Q2
                    if sa_origin in self.q1 and sa_origin in self.q2:
                        self.sa_touched.append(sa_origin)
                        self.q2[sa_origin] = self.q2[sa_origin] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_origin])
                    elif sa_180 in self.q1 and sa_180 in self.q2:
                        self.sa_touched.append(sa_180)
                        self.q2[sa_180] = self.q2[sa_180] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_180])
                    elif sa_lr in self.q1 and sa_lr in self.q2:
                        self.sa_touched.append(sa_lr)
                        self.q2[sa_lr] = self.q2[sa_lr] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_lr])
                    elif sa_180lr in self.q1 and sa_180lr in self.q2:
                        self.sa_touched.append(sa_180lr)
                        self.q2[sa_180lr] = self.q2[sa_180lr] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_180lr])
                    else: 
                        self.q1[sa_origin] = np.random.rand()  
                        self.q2[sa_origin] = np.random.rand()  
                        self.sa_touched.append(sa_origin)
                        self.q2[sa_origin] = self.q2[sa_origin] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_origin])   
            else:       # regular case
                reward = 0

                all_move = []
                for i in range(3):
                    for j in range(3):
                        if board[i][j] == 0:
                            all_move.append((i,j))
                length = len(all_move)
                best_idx = -1
                best_q = -1000

                if np.random.rand() < 0.5:  # update Q1
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
                        if temp_sa_origin in self.q1:
                            if self.q1[temp_sa_origin] > best_q:
                                best_q = self.q1[temp_sa_origin]
                                best_idx = i
                            if temp_sa_origin not in self.q2:
                                self.q2[temp_sa_origin] = np.random.rand()
                                self.sa_touched.append(temp_sa_origin)
                        elif temp_sa_180 in self.q1:
                            if self.q1[temp_sa_180] > best_q:
                                best_q = self.q1[temp_sa_180]
                                best_idx = i
                            if temp_sa_180 not in self.q2:
                                self.q2[temp_sa_180] = np.random.rand()
                                self.sa_touched.append(temp_sa_180)
                        elif temp_sa_lr in self.q1:
                            if self.q1[temp_sa_lr] > best_q:
                                best_q = self.q1[temp_sa_lr]
                                best_idx = i
                            if temp_sa_lr not in self.q2:
                                self.q2[temp_sa_lr] = np.random.rand()
                                self.sa_touched.append(temp_sa_lr)
                        elif temp_sa_180lr in self.q1:
                            if self.q1[temp_sa_180lr] > best_q:
                                best_q = self.q1[temp_sa_180lr]
                                best_idx = i
                            if temp_sa_180lr not in self.q2:
                                self.q2[temp_sa_180lr] = np.random.rand()
                                self.sa_touched.append(temp_sa_180lr)
                        else: 
                            self.q1[temp_sa_origin] = np.random.rand()
                            if self.q1[temp_sa_origin] > best_q:
                                best_q = self.q1[temp_sa_origin]
                                best_idx = i  
                            if temp_sa_origin not in self.q2:
                                self.q2[temp_sa_origin] = np.random.rand()
                            self.sa_touched.append(temp_sa_origin)        
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
                    if new_sa_origin in self.q2:
                        new_sa = new_sa_origin
                    elif new_sa_180 in self.q2:
                        new_sa = new_sa_180
                    elif new_sa_lr in self.q2:
                        new_sa = new_sa_lr
                    elif new_sa_180lr in self.q2:
                        new_sa = new_sa_180lr
                    else:
                        self.q2[new_sa_origin] = np.random.rand()
                        self.sa_touched.append(new_sa_origin)
                        new_sa = new_sa_origin

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
                    if sa_origin in self.q1:
                        self.sa_touched.append(sa_origin)
                        self.q1[sa_origin] = self.q1[sa_origin] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_origin])
                        if sa_origin not in self.q2:
                            self.q2[sa_origin] = np.random.rand()
                    elif sa_180 in self.q1:
                        self.sa_touched.append(sa_180)
                        self.q1[sa_180] = self.q1[sa_180] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_180])
                        if sa_180 not in self.q2:
                            self.q2[sa_180] = np.random.rand()
                    elif sa_lr in self.q1:
                        self.sa_touched.append(sa_lr)
                        self.q1[sa_lr] = self.q1[sa_lr] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_lr])
                        if sa_lr not in self.q2:
                            self.q2[sa_lr] = np.random.rand()
                    elif sa_180lr in self.q1:
                        self.sa_touched.append(sa_180lr)
                        self.q1[sa_180lr] = self.q1[sa_180lr] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_180lr])
                        if sa_180lr not in self.q2:
                            self.q2[sa_180lr] = np.random.rand()
                    else: 
                        self.q1[sa_origin] = np.random.rand()  
                        self.sa_touched.append(sa_origin)
                        self.q1[sa_origin] = self.q1[sa_origin] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_origin])
                        if sa_origin not in self.q2:
                            self.q2[sa_origin] = np.random.rand()
                else:   # update Q2
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
                        if temp_sa_origin in self.q2:
                            if self.q2[temp_sa_origin] > best_q:
                                best_q = self.q2[temp_sa_origin]
                                best_idx = i
                            if temp_sa_origin not in self.q1:
                                self.q1[temp_sa_origin] = np.random.rand()
                                self.sa_touched.append(temp_sa_origin)
                        elif temp_sa_180 in self.q2:
                            if self.q2[temp_sa_180] > best_q:
                                best_q = self.q2[temp_sa_180]
                                best_idx = i
                            if temp_sa_180 not in self.q1:
                                self.q1[temp_sa_180] = np.random.rand()
                                self.sa_touched.append(temp_sa_180)
                        elif temp_sa_lr in self.q2:
                            if self.q2[temp_sa_lr] > best_q:
                                best_q = self.q2[temp_sa_lr]
                                best_idx = i
                            if temp_sa_lr not in self.q1:
                                self.q1[temp_sa_lr] = np.random.rand()
                                self.sa_touched.append(temp_sa_lr)
                        elif temp_sa_180lr in self.q2:
                            if self.q2[temp_sa_180lr] > best_q:
                                best_q = self.q2[temp_sa_180lr]
                                best_idx = i
                            if temp_sa_180lr not in self.q1:
                                self.q1[temp_sa_180lr] = np.random.rand()
                                self.sa_touched.append(temp_sa_180lr)
                        else: 
                            self.q2[temp_sa_origin] = np.random.rand()
                            if self.q2[temp_sa_origin] > best_q:
                                best_q = self.q2[temp_sa_origin]
                                best_idx = i
                            if temp_sa_origin not in self.q1:
                                self.q1[temp_sa_origin] = np.random.rand()
                            self.sa_touched.append(temp_sa_origin)  
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
                    if new_sa_origin in self.q1:
                        new_sa = new_sa_origin
                    elif new_sa_180 in self.q1:
                        new_sa = new_sa_180
                    elif new_sa_lr in self.q1:
                        new_sa = new_sa_lr
                    elif new_sa_180lr in self.q1:
                        new_sa = new_sa_180lr
                    else:
                        self.q1[new_sa_origin] = np.random.rand()
                        self.sa_touched.append(new_sa_origin)
                        new_sa = new_sa_origin

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
                    if sa_origin in self.q2:
                        self.sa_touched.append(sa_origin)
                        self.q2[sa_origin] = self.q2[sa_origin] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_origin])
                        if sa_origin not in self.q1:
                            self.q1[sa_origin] = np.random.rand()
                    elif sa_180 in self.q2:
                        self.sa_touched.append(sa_180)
                        self.q2[sa_180] = self.q2[sa_180] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_180])
                        if sa_180 not in self.q1:
                            self.q1[sa_180] = np.random.rand()
                    elif sa_lr in self.q2:
                        self.sa_touched.append(sa_lr)
                        self.q2[sa_lr] = self.q2[sa_lr] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_lr])
                        if sa_lr not in self.q1:
                            self.q1[sa_lr] = np.random.rand()
                    elif sa_180lr in self.q2:
                        self.sa_touched.append(sa_180lr)
                        self.q2[sa_180lr] = self.q2[sa_180lr] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_180lr])
                        if sa_180lr not in self.q1:
                            self.q1[sa_180lr] = np.random.rand()
                    else: 
                        self.q2[sa_origin] = np.random.rand()  
                        self.sa_touched.append(sa_origin)
                        self.q2[sa_origin] = self.q2[sa_origin] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_origin])
                        if sa_origin not in self.q1:
                            self.q1[sa_origin] = np.random.rand()
        else:   # red just made an action
            if termination:
                if win == 'r':
                    reward = 1
                elif win == 'b':
                    reward = -1
                else:
                    reward = 0

                new_sa_origin = tuple(s_prime_origin + [-1.0, -1.0])
                new_sa_180 = tuple(s_prime_180 + [-1.0, -1.0])
                new_sa_lr = tuple(s_prime_lr + [-1.0, -1.0])
                new_sa_180lr = tuple(s_prime_180lr + [-1.0, -1.0])
                if new_sa_origin in self.q1 and new_sa_origin in self.q2:
                    new_sa = new_sa_origin
                elif new_sa_180 in self.q1 and new_sa_180 in self.q2:
                    new_sa = new_sa_180
                elif new_sa_lr in self.q1 and new_sa_lr in self.q2:
                    new_sa = new_sa_lr
                elif new_sa_180lr in self.q1 and new_sa_180lr in self.q2:
                    new_sa = new_sa_180lr
                else:
                    self.q1[new_sa_origin] = 0
                    self.q2[new_sa_origin] = 0
                    self.sa_touched.append(new_sa_origin)  
                    new_sa = new_sa_origin           
                
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

                if np.random.rand() < 0.5:  # update Q1
                    if sa_origin in self.q1 and sa_origin in self.q2:
                        self.sa_touched.append(sa_origin)
                        self.q1[sa_origin] = self.q1[sa_origin] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_origin])
                    elif sa_180 in self.q1 and sa_180 in self.q2:
                        self.sa_touched.append(sa_180)
                        self.q1[sa_180] = self.q1[sa_180] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_180])
                    elif sa_lr in self.q1 and sa_lr in self.q2:
                        self.sa_touched.append(sa_lr)
                        self.q1[sa_lr] = self.q1[sa_lr] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_lr])
                    elif sa_180lr in self.q1 and sa_180lr in self.q2:
                        self.sa_touched.append(sa_180lr)
                        self.q1[sa_180lr] = self.q1[sa_180lr] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_180lr])
                    else: 
                        self.q1[sa_origin] = np.random.rand()  
                        self.q2[sa_origin] = np.random.rand()  
                        self.sa_touched.append(sa_origin)
                        self.q1[sa_origin] = self.q1[sa_origin] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_origin])
                else:   # update Q2
                    if sa_origin in self.q1 and sa_origin in self.q2:
                        self.sa_touched.append(sa_origin)
                        self.q2[sa_origin] = self.q2[sa_origin] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_origin])
                    elif sa_180 in self.q1 and sa_180 in self.q2:
                        self.sa_touched.append(sa_180)
                        self.q2[sa_180] = self.q2[sa_180] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_180])
                    elif sa_lr in self.q1 and sa_lr in self.q2:
                        self.sa_touched.append(sa_lr)
                        self.q2[sa_lr] = self.q2[sa_lr] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_lr])
                    elif sa_180lr in self.q1 and sa_180lr in self.q2:
                        self.sa_touched.append(sa_180lr)
                        self.q2[sa_180lr] = self.q2[sa_180lr] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_180lr])
                    else: 
                        self.q1[sa_origin] = np.random.rand()  
                        self.q2[sa_origin] = np.random.rand()  
                        self.sa_touched.append(sa_origin)
                        self.q2[sa_origin] = self.q2[sa_origin] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_origin])
    
    def __double_q_update_b(self, board, red_move, termination, win):
        s_prime_origin = board_to_list(board)
        s_prime_180 = board_to_list(board_rotate_180(board))
        s_prime_lr = board_to_list(board_rotate_lr(board))
        s_prime_180lr = board_to_list(board_rotate_180lr(board))
        if red_move:     # red just made an action
            if termination: 
                if win == 'r':
                    reward = -1
                elif win == 'b':
                    reward = 1
                else:
                    reward = 0

                new_sa_origin = tuple(s_prime_origin + [-1.0, -1.0])
                new_sa_180 = tuple(s_prime_180 + [-1.0, -1.0])
                new_sa_lr = tuple(s_prime_lr + [-1.0, -1.0])
                new_sa_180lr = tuple(s_prime_180lr + [-1.0, -1.0])
                if new_sa_origin in self.q1 and new_sa_origin in self.q2:
                    new_sa = new_sa_origin
                elif new_sa_180 in self.q1 and new_sa_180 in self.q2:
                    new_sa = new_sa_180
                elif new_sa_lr in self.q1 and new_sa_lr in self.q2:
                    new_sa = new_sa_lr
                elif new_sa_180lr in self.q1 and new_sa_180lr in self.q2:
                    new_sa = new_sa_180lr
                else:
                    self.q1[new_sa_origin] = 0
                    self.q2[new_sa_origin] = 0
                    self.sa_touched.append(new_sa_origin)  
                    new_sa = new_sa_origin 

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

                if np.random.rand() < 0.5:  # update Q1
                    if sa_origin in self.q1 and sa_origin in self.q2:
                        self.sa_touched.append(sa_origin)
                        self.q1[sa_origin] = self.q1[sa_origin] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_origin])
                    elif sa_180 in self.q1 and sa_180 in self.q2:
                        self.sa_touched.append(sa_180)
                        self.q1[sa_180] = self.q1[sa_180] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_180])
                    elif sa_lr in self.q1 and sa_lr in self.q2:
                        self.sa_touched.append(sa_lr)
                        self.q1[sa_lr] = self.q1[sa_lr] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_lr])
                    elif sa_180lr in self.q1 and sa_180lr in self.q2:
                        self.sa_touched.append(sa_180lr)
                        self.q1[sa_180lr] = self.q1[sa_180lr] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_180lr])
                    else: 
                        self.q1[sa_origin] = np.random.rand()  
                        self.q2[sa_origin] = np.random.rand()  
                        self.sa_touched.append(sa_origin)
                        self.q1[sa_origin] = self.q1[sa_origin] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_origin])
                else:   # update Q2
                    if sa_origin in self.q1 and sa_origin in self.q2:
                        self.sa_touched.append(sa_origin)
                        self.q2[sa_origin] = self.q2[sa_origin] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_origin])
                    elif sa_180 in self.q1 and sa_180 in self.q2:
                        self.sa_touched.append(sa_180)
                        self.q2[sa_180] = self.q2[sa_180] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_180])
                    elif sa_lr in self.q1 and sa_lr in self.q2:
                        self.sa_touched.append(sa_lr)
                        self.q2[sa_lr] = self.q2[sa_lr] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_lr])
                    elif sa_180lr in self.q1 and sa_180lr in self.q2:
                        self.sa_touched.append(sa_180lr)
                        self.q2[sa_180lr] = self.q2[sa_180lr] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_180lr])
                    else: 
                        self.q1[sa_origin] = np.random.rand()  
                        self.q2[sa_origin] = np.random.rand()  
                        self.sa_touched.append(sa_origin)
                        self.q2[sa_origin] = self.q2[sa_origin] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_origin])   
            else:       # regular case
                reward = 0

                all_move = []
                for i in range(3):
                    for j in range(3):
                        if board[i][j] == 0:
                            all_move.append((i,j))
                length = len(all_move)
                best_idx = -1
                best_q = -1000

                if np.random.rand() < 0.5:  # update Q1
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
                        if temp_sa_origin in self.q1:
                            if self.q1[temp_sa_origin] > best_q:
                                best_q = self.q1[temp_sa_origin]
                                best_idx = i
                            if temp_sa_origin not in self.q2:
                                self.q2[temp_sa_origin] = np.random.rand()
                                self.sa_touched.append(temp_sa_origin)
                        elif temp_sa_180 in self.q1:
                            if self.q1[temp_sa_180] > best_q:
                                best_q = self.q1[temp_sa_180]
                                best_idx = i
                            if temp_sa_180 not in self.q2:
                                self.q2[temp_sa_180] = np.random.rand()
                                self.sa_touched.append(temp_sa_180)
                        elif temp_sa_lr in self.q1:
                            if self.q1[temp_sa_lr] > best_q:
                                best_q = self.q1[temp_sa_lr]
                                best_idx = i
                            if temp_sa_lr not in self.q2:
                                self.q2[temp_sa_lr] = np.random.rand()
                                self.sa_touched.append(temp_sa_lr)
                        elif temp_sa_180lr in self.q1:
                            if self.q1[temp_sa_180lr] > best_q:
                                best_q = self.q1[temp_sa_180lr]
                                best_idx = i
                            if temp_sa_180lr not in self.q2:
                                self.q2[temp_sa_180lr] = np.random.rand()
                                self.sa_touched.append(temp_sa_180lr)
                        else: 
                            self.q1[temp_sa_origin] = np.random.rand()
                            if self.q1[temp_sa_origin] > best_q:
                                best_q = self.q1[temp_sa_origin]
                                best_idx = i  
                            if temp_sa_origin not in self.q2:
                                self.q2[temp_sa_origin] = np.random.rand()
                            self.sa_touched.append(temp_sa_origin)        
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
                    if new_sa_origin in self.q2:
                        new_sa = new_sa_origin
                    elif new_sa_180 in self.q2:
                        new_sa = new_sa_180
                    elif new_sa_lr in self.q2:
                        new_sa = new_sa_lr
                    elif new_sa_180lr in self.q2:
                        new_sa = new_sa_180lr
                    else:
                        self.q2[new_sa_origin] = np.random.rand()
                        self.sa_touched.append(new_sa_origin)
                        new_sa = new_sa_origin

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
                    if sa_origin in self.q1:
                        self.sa_touched.append(sa_origin)
                        self.q1[sa_origin] = self.q1[sa_origin] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_origin])
                        if sa_origin not in self.q2:
                            self.q2[sa_origin] = np.random.rand()
                    elif sa_180 in self.q1:
                        self.sa_touched.append(sa_180)
                        self.q1[sa_180] = self.q1[sa_180] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_180])
                        if sa_180 not in self.q2:
                            self.q2[sa_180] = np.random.rand()
                    elif sa_lr in self.q1:
                        self.sa_touched.append(sa_lr)
                        self.q1[sa_lr] = self.q1[sa_lr] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_lr])
                        if sa_lr not in self.q2:
                            self.q2[sa_lr] = np.random.rand()
                    elif sa_180lr in self.q1:
                        self.sa_touched.append(sa_180lr)
                        self.q1[sa_180lr] = self.q1[sa_180lr] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_180lr])
                        if sa_180lr not in self.q2:
                            self.q2[sa_180lr] = np.random.rand()
                    else: 
                        self.q1[sa_origin] = np.random.rand()  
                        self.sa_touched.append(sa_origin)
                        self.q1[sa_origin] = self.q1[sa_origin] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_origin])
                        if sa_origin not in self.q2:
                            self.q2[sa_origin] = np.random.rand()
                else:   # update Q2
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
                        if temp_sa_origin in self.q2:
                            if self.q2[temp_sa_origin] > best_q:
                                best_q = self.q2[temp_sa_origin]
                                best_idx = i
                            if temp_sa_origin not in self.q1:
                                self.q1[temp_sa_origin] = np.random.rand()
                                self.sa_touched.append(temp_sa_origin)
                        elif temp_sa_180 in self.q2:
                            if self.q2[temp_sa_180] > best_q:
                                best_q = self.q2[temp_sa_180]
                                best_idx = i
                            if temp_sa_180 not in self.q1:
                                self.q1[temp_sa_180] = np.random.rand()
                                self.sa_touched.append(temp_sa_180)
                        elif temp_sa_lr in self.q2:
                            if self.q2[temp_sa_lr] > best_q:
                                best_q = self.q2[temp_sa_lr]
                                best_idx = i
                            if temp_sa_lr not in self.q1:
                                self.q1[temp_sa_lr] = np.random.rand()
                                self.sa_touched.append(temp_sa_lr)
                        elif temp_sa_180lr in self.q2:
                            if self.q2[temp_sa_180lr] > best_q:
                                best_q = self.q2[temp_sa_180lr]
                                best_idx = i
                            if temp_sa_180lr not in self.q1:
                                self.q1[temp_sa_180lr] = np.random.rand()
                                self.sa_touched.append(temp_sa_180lr)
                        else: 
                            self.q2[temp_sa_origin] = np.random.rand()
                            if self.q2[temp_sa_origin] > best_q:
                                best_q = self.q2[temp_sa_origin]
                                best_idx = i
                            if temp_sa_origin not in self.q1:
                                self.q1[temp_sa_origin] = np.random.rand()
                            self.sa_touched.append(temp_sa_origin)  
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
                    if new_sa_origin in self.q1:
                        new_sa = new_sa_origin
                    elif new_sa_180 in self.q1:
                        new_sa = new_sa_180
                    elif new_sa_lr in self.q1:
                        new_sa = new_sa_lr
                    elif new_sa_180lr in self.q1:
                        new_sa = new_sa_180lr
                    else:
                        self.q1[new_sa_origin] = np.random.rand()
                        self.sa_touched.append(new_sa_origin)
                        new_sa = new_sa_origin

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
                    if sa_origin in self.q2:
                        self.sa_touched.append(sa_origin)
                        self.q2[sa_origin] = self.q2[sa_origin] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_origin])
                        if sa_origin not in self.q1:
                            self.q1[sa_origin] = np.random.rand()
                    elif sa_180 in self.q2:
                        self.sa_touched.append(sa_180)
                        self.q2[sa_180] = self.q2[sa_180] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_180])
                        if sa_180 not in self.q1:
                            self.q1[sa_180] = np.random.rand()
                    elif sa_lr in self.q2:
                        self.sa_touched.append(sa_lr)
                        self.q2[sa_lr] = self.q2[sa_lr] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_lr])
                        if sa_lr not in self.q1:
                            self.q1[sa_lr] = np.random.rand()
                    elif sa_180lr in self.q2:
                        self.sa_touched.append(sa_180lr)
                        self.q2[sa_180lr] = self.q2[sa_180lr] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_180lr])
                        if sa_180lr not in self.q1:
                            self.q1[sa_180lr] = np.random.rand()
                    else: 
                        self.q2[sa_origin] = np.random.rand()  
                        self.sa_touched.append(sa_origin)
                        self.q2[sa_origin] = self.q2[sa_origin] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_origin])
                        if sa_origin not in self.q1:
                            self.q1[sa_origin] = np.random.rand()
        else:   # black just made an action
            if termination:
                if win == 'r':
                    reward = -1
                elif win == 'b':
                    reward = 1
                else:
                    reward = 0

                new_sa_origin = tuple(s_prime_origin + [-1.0, -1.0])
                new_sa_180 = tuple(s_prime_180 + [-1.0, -1.0])
                new_sa_lr = tuple(s_prime_lr + [-1.0, -1.0])
                new_sa_180lr = tuple(s_prime_180lr + [-1.0, -1.0])
                if new_sa_origin in self.q1 and new_sa_origin in self.q2:
                    new_sa = new_sa_origin
                elif new_sa_180 in self.q1 and new_sa_180 in self.q2:
                    new_sa = new_sa_180
                elif new_sa_lr in self.q1 and new_sa_lr in self.q2:
                    new_sa = new_sa_lr
                elif new_sa_180lr in self.q1 and new_sa_180lr in self.q2:
                    new_sa = new_sa_180lr
                else:
                    self.q1[new_sa_origin] = 0
                    self.q2[new_sa_origin] = 0
                    self.sa_touched.append(new_sa_origin)  
                    new_sa = new_sa_origin           
                
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

                if np.random.rand() < 0.5:  # update Q1
                    if sa_origin in self.q1 and sa_origin in self.q2:
                        self.sa_touched.append(sa_origin)
                        self.q1[sa_origin] = self.q1[sa_origin] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_origin])
                    elif sa_180 in self.q1 and sa_180 in self.q2:
                        self.sa_touched.append(sa_180)
                        self.q1[sa_180] = self.q1[sa_180] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_180])
                    elif sa_lr in self.q1 and sa_lr in self.q2:
                        self.sa_touched.append(sa_lr)
                        self.q1[sa_lr] = self.q1[sa_lr] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_lr])
                    elif sa_180lr in self.q1 and sa_180lr in self.q2:
                        self.sa_touched.append(sa_180lr)
                        self.q1[sa_180lr] = self.q1[sa_180lr] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_180lr])
                    else: 
                        self.q1[sa_origin] = np.random.rand()  
                        self.q2[sa_origin] = np.random.rand()  
                        self.sa_touched.append(sa_origin)
                        self.q1[sa_origin] = self.q1[sa_origin] + self.lr*(reward + self.gamma*self.q2[new_sa] - self.q1[sa_origin])
                else:   # update Q2
                    if sa_origin in self.q1 and sa_origin in self.q2:
                        self.sa_touched.append(sa_origin)
                        self.q2[sa_origin] = self.q2[sa_origin] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_origin])
                    elif sa_180 in self.q1 and sa_180 in self.q2:
                        self.sa_touched.append(sa_180)
                        self.q2[sa_180] = self.q2[sa_180] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_180])
                    elif sa_lr in self.q1 and sa_lr in self.q2:
                        self.sa_touched.append(sa_lr)
                        self.q2[sa_lr] = self.q2[sa_lr] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_lr])
                    elif sa_180lr in self.q1 and sa_180lr in self.q2:
                        self.sa_touched.append(sa_180lr)
                        self.q2[sa_180lr] = self.q2[sa_180lr] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_180lr])
                    else: 
                        self.q1[sa_origin] = np.random.rand()  
                        self.q2[sa_origin] = np.random.rand()  
                        self.sa_touched.append(sa_origin)
                        self.q2[sa_origin] = self.q2[sa_origin] + self.lr*(reward + self.gamma*self.q1[new_sa] - self.q2[sa_origin])
