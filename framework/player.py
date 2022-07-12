import copy
import yaml
import random
import numpy as np
import torch
from os.path import exists
from heapq import heapify, heappop, heappush
from framework.utils import board_turn180, rotate_action, board_trans
from framework.utils import board_rotate_90, board_rotate_180, board_rotate_270, board_rotate_ud, board_rotate_lr
from framework.constant import piece_values, action2posi
from ai.network import DQN
from min_max_test.board_test import Board
from min_max_test.player_test import aiPlayer

class Player():
    def __init__(self, color):
        self.color = color
        self.faction = 1
        if color == 'b':
            self.faction = -1
        self.current_board = None
        self.current_piece_value = 0
        self.current_piece_posi = None
        self.all_move = {}
    
    def check_moves(self):
        self.all_move = {}
        for i in range(3):
            for j in range(3):
                if self.current_board[i][j] == 0:
                    self.all_move[(i,j)] = self.faction
        if len(self.all_move) > 0:
            return True
        else:
            return False
    
    def update_board(self, board):
        self.current_board = copy.deepcopy(board)

class HumanPlayer(Player):
    def __init__(self, color):
        self.color = color
        self.faction = 1
        if color == 'b':
            self.faction = -1
        self.current_board = None
        self.current_piece_value = 0
        self.current_piece_posi = None
        self.all_move = {}
        self.move = None

    def reset(self):
        self.current_board = None
        self.current_piece_value = 0
        self.current_piece_posi = None
        self.all_move = {}
        self.move = None

    def human_action(self, posi):
        if posi in self.all_move:
            return posi, self.faction
        else:
            return None, None

class AIPlayer(Player):
    def __init__(self, color, explore_rate=1):
        self.color = color
        self.explore_rate = explore_rate
        self.faction = 1
        self.current_board = None
        self.current_piece_value = 0
        self.current_piece_posi = None
        self.all_move = {}
        with open('ai/config.yaml') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        if not exists(self.config['final_model_path']):  # random action only
            self.explore_rate = 1
        if self.explore_rate < 1:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.q_star = DQN().to(self.device)
            checkpoint = torch.load(self.config['final_model_path'])
            self.q_star.load_state_dict(checkpoint['model_state_dict'])
            self.q_star.eval()

    def reset(self):
        self.current_board = None
        self.current_piece_value = 0
        self.current_piece_posi = None
        self.all_move = {}
        # if self.explore_rate < 1:
        #     checkpoint = torch.load(self.config['final_model_path'])
        #     self.q_star.load_state_dict(checkpoint['model_state_dict'])
        #     self.q_star.eval()

    def set_explore_rate(self, explore_rate):
        self.explore_rate = explore_rate
        if not exists(self.config['final_model_path']):  # random action only
            self.explore_rate = 1

    def update_board(self, board):
        if self.color == 'b':       # rotate board
            self.current_board = board_turn180(board)
        else:
            self.current_board = board

    def __random_action(self):
        posi_num = len(self.all_move)
        posi_idx = np.random.randint(posi_num)
        idx = 0
        for key in self.all_move:
            if idx == posi_idx:
                posi = key
                break
            idx += 1
        value = self.all_move[posi]
        if self.color == 'b':       # rotate move
            return rotate_action(posi, value)
        return posi, value

    def __exploit_action(self):
        temp = board_trans(copy.deepcopy(self.current_board))
        temp_all = np.zeros((6,3,3))
        temp_all[0] = temp
        temp_all[1] = board_rotate_90(temp)
        temp_all[2] = board_rotate_180(temp)
        temp_all[3] = board_rotate_270(temp)
        temp_all[4] = board_rotate_ud(temp)
        temp_all[5] = board_rotate_lr(temp)
        state = torch.from_numpy(temp_all).to(self.device)
        if str(self.device) == 'cuda':
            state = state.float().view(1, 6, 3, 3).type(torch.cuda.FloatTensor)
        else:
            state = state.float().view(1, 6, 3, 3).type(torch.FloatTensor)
        queue = []
        heapify(queue)
        for posi in self.all_move:
            next_board = copy.deepcopy(self.current_board)
            next_board[posi[0]][posi[1]] = 1
            temp = board_trans(next_board)
            temp_all = np.zeros((6,3,3))
            temp_all[0] = temp
            temp_all[1] = board_rotate_90(temp)
            temp_all[2] = board_rotate_180(temp)
            temp_all[3] = board_rotate_270(temp)
            temp_all[4] = board_rotate_ud(temp)
            temp_all[5] = board_rotate_lr(temp)
            action = torch.from_numpy(temp_all).to(self.device)
            if str(self.device) == 'cuda':
                action = action.float().view(1, 6, 3, 3).type(torch.cuda.FloatTensor)
            else:
                action = action.float().view(1, 6, 3, 3).type(torch.FloatTensor)
            win_rate = self.q_star(state, action)
            win_rate = win_rate.cpu().detach().numpy()[0][0]
            heappush(queue, (-win_rate, (posi, 1)))
        value, action = heappop(queue)
        if self.color == 'b':       # rotate move
            return rotate_action(action[0], action[1])
        return action[0], action[1]
    
    def ai_action(self):
        if self.explore_rate == 0:
            return self.__exploit_action()
        elif self.explore_rate == 1:
            return self.__random_action()
        else:
            sample = random.random()
            if sample > self.explore_rate:
                return self.__exploit_action()
            else:
                return self.__random_action()



class MinMaxPlayer(Player):
    def __init__(self, color):
        self.color = color
        self.faction = 1
        self.player = aiPlayer('X')
        if color == 'b':
            self.faction = -1
            self.player = aiPlayer('O')
        self.current_board = None
        self.current_piece_value = 0
        self.current_piece_posi = None
        self.all_move = {}
        
    
    def reset(self):
        self.current_board = None
        self.current_piece_value = 0
        self.current_piece_posi = None
        self.all_move = {}

    def minmax_action(self):
        board = Board()
        idx = 0
        for i in range(3):
            for j in range(3):
                if self.current_board[i][j] == 0:
                    board._board[idx] = '-'
                if self.current_board[i][j] == 1:
                    board._board[idx] = 'X'
                if self.current_board[i][j] == -1:
                    board._board[idx] = 'O'
                idx += 1

        action = self.player.think(board)
        posi = action2posi[action]
        return posi, self.faction
    
    def ai_action(self):
        return self.minmax_action()

    # def minmax_action(self):
    #     if self.color == 'b':       # rotate board
    #         op_color = 'r'
    #     else:
    #         op_color = 'b'
    #     op_player = MinMaxPlayer(op_color) # 假想的敌人
    #     chess_board = ChessBoard()
    #     chess_board.load_board(self.current_board)
    #     _,posi =self.minimax(chess_board, op_player)
    #     return posi, self.faction

    # def minimax(self, board, op_player, depth=0):
    #     if self.faction == 'b':
    #         bestVal = -10
    #     else:
    #         bestVal = 10

    #     board.check_done()
    #     if board.done:
    #         # print('GET HERE!!!!!!!!!!!!!!!')
    #         if board.win == 'r':
    #             # "X"胜利
    #             return -10 + depth, None
    #         elif board.win == 'b':
    #             # "O"胜利
    #             return 10 - depth, None
    #         else:
    #             # 平局
    #             return 0,None
        
    #     # 遍历合法走法 action = posi
    #     all_move = board.get_legal_actions(self.color)
    #     # for key in self.all_move:
    #     #     bestAction = key
    #     # print(len(all_move))
    #     for action in all_move:
    #         board.move_piece(action, self.faction, record = False)
    #         # print(action)
    #         val,_ = op_player.minimax(board,self,depth+1) # 切换到假想敌
    #         print('val = ' + str(val))
    #         print('depth = ' + str(depth))
    #         board.unmove(action) # 撤销走法，回溯

    #         if self.faction == 1:
    #             if val >= bestVal: # Max
    #                 bestVal,bestAction = val,action
    #                 # print('GET HERE!')
    #         else: # Min
    #             if val <= bestVal:
    #                 bestVal,bestAction = val,action
    #                 # print('GET HERE!!')
    #     return bestVal,bestAction