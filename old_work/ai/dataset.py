import h5py
import numpy as np
from torch.utils.data import Dataset
from framework.utils import key_to_board

class HDF5_Dataset(Dataset):
    def __init__(self, h5py_x_path='dataset/x.hdf5', h5py_y_path='dataset/y.hdf5'):
        self.board = []
        self.action = []
        self.win_rate = []
        fx = h5py.File(h5py_x_path, 'r')
        fy = h5py.File(h5py_y_path, 'r')
        for key in fx:
            board = np.array(fx[key])[0]
            action = np.array(fx[key])[1]
            data = np.array(fy[key])
            win_rate = float(data[0]) / (data[0]+data[1]+data[2])
            self.board.append(board)
            self.action.append(action)
            self.win_rate.append(win_rate)
        fx.close()
        fy.close()
        self.length = len(self.board)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {'board': self.board[idx],
                  'action': self.action[idx],
                  'win_rate': self.win_rate[idx]}
        return sample

class DICT_Dataset(Dataset):
    def __init__(self, dataset_dict):
        self.board = []
        self.action = []
        self.win_rate = []
        for key in dataset_dict:
            board = key_to_board(key[0])
            action = key_to_board(key[1])
            data = dataset_dict[key]
            win_rate = float(data[0]) / (data[0]+data[1]+data[2])
            self.board.append(board)
            self.action.append(action)
            self.win_rate.append(win_rate)
        self.length = len(self.board)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {'board': self.board[idx],
                  'action': self.action[idx],
                  'win_rate': self.win_rate[idx]}
        return sample