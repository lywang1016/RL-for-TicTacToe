import numpy as np
import h5py
from os.path import exists

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

def board_to_list(board):
    key = []
    for i in range(3):
        for j in range(3):
            key.append(board[i][j])
    return key

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

def merge_dataset(dataset1, dataset2): # merge dataset2 data into dataset1
    for key in dataset2:
        if key not in dataset1:
            dataset1[key] = dataset2[key]
        else:
            for i in range(3):
                dataset1[key][i] += dataset2[key][i]

# def h5py_to_dataset(x_path, y_path):
#     dataset = {}
#     if exists(x_path) and exists(y_path):
#         fx = h5py.File(x_path, 'r')
#         fy = h5py.File(y_path, 'r')
#         for key in fx:
#             boards = np.array(fx[key])
#             data = np.array(fy[key])
#             dataset[(board_to_key(boards[0]), board_to_key(boards[1]))] = data
#         fx.close()
#         fy.close()
#     return dataset

def dataset_to_h5py(dataset, x_path, y_path):
    fx = h5py.File(x_path, "w")
    fy = h5py.File(y_path, "w")
    for key in dataset:
        fx.create_dataset(str(key), data=np.array((key_to_board(key[0]), key_to_board(key[1]))))
        fy.create_dataset(str(key), data=dataset[key])
    fx.close()
    fy.close()

def h5py_add_dataset(x_path, y_path, dataset):
    fx = h5py.File(x_path, "a")
    fy = h5py.File(y_path, "a")
    for key in dataset:
        if str(key) in fx:
            data_in_file = np.array(fy[str(key)])
            for i in range(3):
                data_in_file[i] += dataset[key][i]
            del fy[str(key)]
            fy.create_dataset(str(key), data=data_in_file)
        else:
            fx.create_dataset(str(key), data=np.array((key_to_board(key[0]), key_to_board(key[1]))))
            fy.create_dataset(str(key), data=dataset[key])
    fx.close()
    fy.close()
