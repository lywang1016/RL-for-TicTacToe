import numpy as np

def board_to_list(board):
    key = []
    for i in range(3):
        for j in range(3):
            key.append(board[i][j])
    return key

def board_rotate_90(board_in):
    board = np.zeros((3, 3))
    for i in range(3):
        board[:,i] = board_in[2-i,:]
    return board

def board_rotate_180(board_in):
    board = board_in[::-1,::-1]
    return board

def board_rotate_270(board_in):
    board = np.zeros((3, 3))
    for i in range(3):
        board[i,:] = board_in[:,2-i]
    return board

def board_rotate_ud(board_in):
    board = np.zeros((3, 3))
    board[0,:] = board_in[2,:]
    board[1,:] = board_in[1,:]
    board[2,:] = board_in[0,:]
    return board

def board_rotate_lr(board_in):
    board = np.zeros((3, 3))
    board[:,0] = board_in[:,2]
    board[:,1] = board_in[:,1]
    board[:,2] = board_in[:,0]
    return board

def board_trans(board):
    res = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            if board[i][j] >= 0:
                res[i][j] = board[i][j]
            else:
                res[i][j] = 2
    return res

def key_to_board(key):
    board = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            board[i][j] = key[i][j]
    return board

def rotate_action(posi, value):
    value_ = -value
    posi_ = (2 - posi[0], 2 - posi[1])
    return posi_, value_

def board_turn180(board_in):
    board = board_in[::-1,::-1]
    for i in range(3):
        for j in range(3):
            board[i][j] = -board[i][j]
    return board
