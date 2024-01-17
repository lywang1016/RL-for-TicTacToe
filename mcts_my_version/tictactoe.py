import numpy as np
from utils import piece_values, action_change_perspective

class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count
        self.actions_encode = {}
        self.actions_decode = {}
        self.actions_encode[(0,0)] = 0
        self.actions_decode[0] = (0,0)
        self.actions_encode[(0,1)] = 1
        self.actions_decode[1] = (0,1)
        self.actions_encode[(0,2)] = 2
        self.actions_decode[2] = (0,2)
        self.actions_encode[(1,0)] = 3
        self.actions_decode[3] = (1,0)
        self.actions_encode[(1,1)] = 4
        self.actions_decode[4] = (1,1)
        self.actions_encode[(1,2)] = 5
        self.actions_decode[5] = (1,2)
        self.actions_encode[(2,0)] = 6
        self.actions_decode[6] = (2,0)
        self.actions_encode[(2,1)] = 7
        self.actions_decode[7] = (2,1)
        self.actions_encode[(2,2)] = 8
        self.actions_decode[8] = (2,2)

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):        # state in view of player 1 and player is going to play
        if player == 1:
            posi = self.actions_decode[action]
            row = posi[0]
            column = posi[1]
            state[row, column] = 1
            return state
        else:
            posi = self.actions_decode[action]
            posi = action_change_perspective(posi)
            row = posi[0]
            column = posi[1]
            state[row, column] = -1
            return state
    
    def get_valid_moves(self, state, player):               # state in view of player 1 and player is going to play
        state = self.change_perspective(state.copy(), player)
        valid_moves = []
        for i in range(self.row_count):
            for j in range(self.column_count):
                if state[i][j] == 0:
                    action = self.actions_encode[(i,j)]
                    valid_moves.append(action)
        return valid_moves
    
    def valid_moves_mask(self, valid_moves):
        mask = []
        for i in range(self.action_size):
            if i in valid_moves:
                mask.append(1)
            else:
                mask.append(0)
        return np.array(mask, dtype=np.uint8)
    
    def get_value_and_terminated(self, state, player):      # state in view of player 1 and player just played
        for i in range(3):
            if state[i][0]==state[i][1] and state[i][1]==state[i][2]:
                if state[i][0]==piece_values['r_piece']:
                    if player == 1:
                        return 1, True
                    else:
                        return 0, True
                if state[i][0]==piece_values['b_piece']:
                    if player == -1:
                        return 1, True
                    else:
                        return 0, True
        for i in range(3):
            if state[0][i]==state[1][i] and state[1][i]==state[2][i]:
                if state[0][i]==piece_values['r_piece']:
                    if player == 1:
                        return 1, True
                    else:
                        return 0, True
                if state[0][i]==piece_values['b_piece']:
                    if player == -1:
                        return 1, True
                    else:
                        return 0, True
        v1 = state[0][0]
        v2 = state[1][1]
        v3 = state[2][2]
        if v1==v2 and v2==v3:
            if v1==piece_values['r_piece']:
                if player == 1:
                    return 1, True
                else:
                    return 0, True
            if v1==piece_values['b_piece']:
                if player == -1:
                    return 1, True
                else:
                    return 0, True
        v1 = state[0][2]
        v3 = state[2][0]
        if v1==v2 and v2==v3:
            if v1==piece_values['r_piece']:
                if player == 1:
                    return 1, True
                else:
                    return 0, True
            if v1==piece_values['b_piece']:
                if player == -1:
                    return 1, True
                else:
                    return 0, True
        if_zero = False
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    if_zero = True
        if if_zero:
            return 0.5, False
        else:
            return 0.5, True
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return 1-value
    
    def change_perspective(self, state, player):
        if player > 0:
            return state
        else:
            board = state[::-1,::-1]
            for i in range(3):
                for j in range(3):
                    board[i][j] = -board[i][j]
            return board


