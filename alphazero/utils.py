import numpy as np

piece_values = {
    'b_piece': -1,
    'r_piece': 1
}

values_piece = {
    -1: 'b_piece',
    1: 'r_piece'
}

posi_idx_map = {
    (-1, -1): 0,
    (0, 0): 1,
    (0, 1): 2,
    (0, 2): 3,
    (1, 0): 4,
    (1, 1): 5,
    (1, 2): 6,
    (2, 0): 7,
    (2, 1): 8,
    (2, 2): 9
}

idx_rotate_180 = {
    0: (-1, -1),
    1: (2, 2),
    2: (2, 1),
    3: (2, 0),
    4: (1, 2),
    5: (1, 1),
    6: (1, 0),
    7: (0, 2),
    8: (0, 1),
    9: (0, 0)
}

idx_rotate_lr = {
    0: (-1, -1),
    1: (0, 2),
    2: (0, 1),
    3: (0, 0),
    4: (1, 2),
    5: (1, 1),
    6: (1, 0),
    7: (2, 2),
    8: (2, 1),
    9: (2, 0)
}

idx_rotate_180lr = {
    0: (-1, -1),
    1: (2, 0),
    2: (2, 1),
    3: (2, 2),
    4: (1, 0),
    5: (1, 1),
    6: (1, 2),
    7: (0, 0),
    8: (0, 1),
    9: (0, 2)
}

def action_change_perspective(posi):
    posi_new = (2-posi[0], 2-posi[1])
    return posi_new

def board_to_list(board):
    key = []
    for i in range(3):
        for j in range(3):
            key.append(board[i][j])
    return key

def board_rotate_lr(board_in):
    board = np.zeros((3, 3))
    board[:,0] = board_in[:,2]
    board[:,1] = board_in[:,1]
    board[:,2] = board_in[:,0]
    return board

def board_rotate_180(board_in):
    board = board_in[::-1,::-1]
    return board

def board_rotate_180lr(board_in):
    board = board_rotate_180(board_in)
    return board_rotate_lr(board)

if __name__ == '__main__':
    board = np.zeros((3, 3))
    board[0][0] = 1
    board[0][1] = 2
    board[0][2] = 3
    board[1][0] = 4
    board[1][1] = 5
    board[1][2] = 6
    board[2][0] = 7
    board[2][1] = 8
    board[2][2] = 9
    print(board)
    board1 = board_rotate_lr(board)
    print(board1)
    board2 = board_rotate_180(board)
    print(board2)
    board3 = board_rotate_180lr(board)
    print(board3)