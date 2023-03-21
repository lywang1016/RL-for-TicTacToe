import copy
import numpy as np
from constant import piece_values

class ChessBoard:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.red = True
        self.done = False
        self.win = None
        self.reset_board()

    def reset_board(self):
        self.board = np.zeros((3, 3))
        self.done = False
        self.red = True
        self.win = None

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
        if self.board[position[0]][position[1]] == 0:
            self.board[position[0]][position[1]] = value
            self.check_done()
            self.red = not self.red

    