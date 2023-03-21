import copy
import csv
import datetime
import numpy as np
from framework.utils import board_turn180, board_to_key
from framework.constant import piece_values

class ChessBoard:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.red = True
        self.done = False
        self.win = None
        self.dataset = {}
        self.red_history = []
        self.black_history = []
        self.action_history = []
        self.red_action_board = []
        self.black_action_board = []
        self.reset_board()

    def reset_board(self):
        self.board = np.zeros((3, 3))
        self.done = False
        self.red = True
        self.win = None
        self.dataset = {}
        self.red_history = []
        self.black_history = []
        self.dataset = {}
        self.action_history = []
        self.red_action_board = []
        self.black_action_board = []

    def set_done(self, win_color):
        self.win = win_color
        self.done = True

    def check_done(self):
        for i in range(3):
            if self.board[i][0]==self.board[i][1] and self.board[i][1]==self.board[i][2]:
                if self.board[i][0]==piece_values['r_piece']:
                    self.win = 'r'
                    self.done = True
                    return
                if self.board[i][0]==piece_values['b_piece']:
                    self.win = 'b'
                    self.done = True
                    return
        for i in range(3):
            if self.board[0][i]==self.board[1][i] and self.board[1][i]==self.board[2][i]:
                if self.board[0][i]==piece_values['r_piece']:
                    self.win = 'r'
                    self.done = True
                    return
                if self.board[0][i]==piece_values['b_piece']:
                    self.win = 'b'
                    self.done = True
                    return
        v1 = self.board[0][0]
        v2 = self.board[1][1]
        v3 = self.board[2][2]
        if v1==v2 and v2==v3:
            if v1==piece_values['r_piece']:
                self.win = 'r'
                self.done = True
                return
            if v1==piece_values['b_piece']:
                self.win = 'b'
                self.done = True
                return
        v1 = self.board[0][2]
        v3 = self.board[2][0]
        if v1==v2 and v2==v3:
            if v1==piece_values['r_piece']:
                self.win = 'r'
                self.done = True
                return
            if v1==piece_values['b_piece']:
                self.win = 'b'
                self.done = True
                return
        if_zero = False
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    if_zero = True
        if if_zero:
            self.win = None
            self.done = False
        else:
            self.win = 't'
            self.done = True

    def load_board(self, board):
        self.board = board
        self.check_done()
    
    def board_states(self):
        return copy.deepcopy(self.board)
    
    def move_piece(self, position, value):
        self.action_history.append((position[0], position[1], value))

        if self.red:
            self.red_history.append(board_to_key(self.board_states()))
        else:
            self.black_history.append(board_to_key(board_turn180(self.board_states())))

        if self.board[position[0]][position[1]] == 0:
            self.board[position[0]][position[1]] = value
        self.check_done()

        if self.red:
            self.red_action_board.append(board_to_key(self.board_states()))
        else:
            self.black_action_board.append(board_to_key(board_turn180(self.board_states())))
        self.red = not self.red

    def fill_dataset(self):
        red_len = len(self.red_history)
        black_len = len(self.black_history)
        if self.win == 'r':
            for i in range(red_len):
                key = (self.red_history[i], self.red_action_board[i])
                if key not in self.dataset:
                    self.dataset[key] = [1, 0, 0]
                else:
                    self.dataset[key][0] += 1
            for i in range(black_len):
                key = (self.black_history[i], self.black_action_board[i])
                if key not in self.dataset:
                    self.dataset[key] = [0, 0, 1]
                else:
                    self.dataset[key][2] += 1
        if self.win == 'b':
            for i in range(black_len):
                key = (self.black_history[i], self.black_action_board[i])
                if key not in self.dataset:
                    self.dataset[key] = [1, 0, 0]
                else:
                    self.dataset[key][0] += 1
            for i in range(red_len):
                key = (self.red_history[i], self.red_action_board[i])
                if key not in self.dataset:
                    self.dataset[key] = [0, 0, 1]
                else:
                    self.dataset[key][2] += 1
        if self.win == 't':
            for i in range(red_len):
                key = (self.red_history[i], self.red_action_board[i])
                if key not in self.dataset:
                    self.dataset[key] = [0, 1, 0]
                else:
                    self.dataset[key][1] += 1
            for i in range(black_len):
                key = (self.black_history[i], self.black_action_board[i])
                if key not in self.dataset:
                    self.dataset[key] = [0, 1, 0]
                else:
                    self.dataset[key][1] += 1

    def save_csv(self):
        time_info = datetime.datetime.now()
        file_name = str(time_info.year)+'_'+str(time_info.month)+'_'+str(time_info.day)+'-'\
                    +str(time_info.hour)+'_'+str(time_info.minute)+'_'+str(time_info.second)+'_'\
                    +str(time_info.microsecond)+'.csv'
        f = open('game_record/'+file_name, 'w', newline='')
        writer = csv.writer(f)
        for action in self.action_history:
            row = [action[0], action[1], action[2]]
            writer.writerow(row)
        f.close()