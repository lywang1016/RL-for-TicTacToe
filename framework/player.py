import copy
import yaml
import random
import numpy as np
import torch
from os.path import exists
from heapq import heapify, heappop, heappush
from framework.utils import board_turn180, board_to_key
from framework.constant import piece_values
# from ai.network import DQN

class Player():
    def __init__(self, color):
        self.color = color
        self.faction = 1
        if color == 'b':
            self.faction = -1
        self.current_board = None
        self.current_piece_value = 0
        self.current_piece_posi = None
        self.cadidate_move = []
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
        self.cadidate_move = []
        self.all_move = {}
        self.move = None

    def reset(self):
        self.current_board = None
        self.current_piece_value = 0
        self.current_piece_posi = None
        self.cadidate_move = []
        self.all_move = {}
        self.move = None

    def human_action(self, posi):
        if posi in self.all_move:
            return posi, self.faction
        else:
            return None, None

# class AIPlayer(Player):
#     def __init__(self, color, explore_rate=1):
#         self.color = color
#         self.explore_rate = explore_rate
#         self.faction = 1
#         self.current_board = None
#         self.current_piece_value = 0
#         self.current_piece_posi = None
#         self.cadidate_move = []
#         self.all_move = {}
#         self.past_board = []
#         self.past_actions = []
#         with open('ai/config.yaml') as f:
#             self.config = yaml.load(f, Loader=yaml.FullLoader)
#         if not exists(self.config['save_model_path']):  # random action only
#             self.explore_rate = 1
#         if self.explore_rate < 1:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             self.q_star = DQN().to(self.device)
#             checkpoint = torch.load(self.config['save_model_path'])
#             self.q_star.load_state_dict(checkpoint['model_state_dict'])
#             self.q_star.eval()

#     def reset(self):
#         self.current_board = None
#         self.current_piece_value = 0
#         self.current_piece_posi = None
#         self.cadidate_move = []
#         self.all_move = {}
#         self.past_board = []
#         self.past_actions = []
#         if self.explore_rate < 1:
#             checkpoint = torch.load(self.config['save_model_path'])
#             self.q_star.load_state_dict(checkpoint['model_state_dict'])
#             self.q_star.eval()

#     def update_board(self, board, past_board, past_actions):
#         if self.color == 'b':       # rotate board
#             self.current_board = board_turn180(board)
#         else:
#             self.current_board = board
#         self.past_board = past_board
#         self.past_actions = past_actions

#     def check_moves(self):
#         self.all_move = {}
#         for i in range(10):
#             for j in range(9):
#                 if self.current_board[i][j] * self.faction > 0:
#                     value = abs(self.current_board[i][j])
#                     if value == 1:
#                         rook = Rook('r', (i, j))
#                         moves = rook.next_valid_move(self.current_board)
#                         if len(moves) > 0:
#                             self.all_move[(i,j)] = moves
#                     if value == 2:
#                         knight = Knight('r', (i, j))
#                         moves = knight.next_valid_move(self.current_board)
#                         if len(moves) > 0:
#                             self.all_move[(i,j)] = moves
#                     if value == 3:
#                         cannon = Cannon('r', (i, j))
#                         moves = cannon.next_valid_move(self.current_board)
#                         if len(moves) > 0:
#                             self.all_move[(i,j)] = moves
#                     if value == 4:
#                         minister = Minister('r', (i, j))
#                         moves = minister.next_valid_move(self.current_board)
#                         if len(moves) > 0:
#                             self.all_move[(i,j)] = moves
#                     if value == 5:
#                         warrior = Warrior('r', (i, j))
#                         moves = warrior.next_valid_move(self.current_board)
#                         if len(moves) > 0:
#                             self.all_move[(i,j)] = moves
#                     if value == 6:
#                         pawn = Pawn('r', (i, j))
#                         moves = pawn.next_valid_move(self.current_board)
#                         if len(moves) > 0:
#                             self.all_move[(i,j)] = moves
#                     if value == 7:
#                         king = King('r', (i, j))
#                         moves = king.next_valid_move(self.current_board)
#                         if len(moves) > 0:
#                             self.all_move[(i,j)] = moves
#         for posi in self.all_move: # if able to take black king, the only valid action is take the king
#             for move in self.all_move[posi]:
#                 if self.current_board[posi[0]+move[0]][posi[1]+move[1]] == piece_values['b_king']:
#                     self.all_move = {}
#                     self.all_move[posi] = [move]
#                     return True

#         current_key = board_to_key(copy.deepcopy(self.current_board)) # remove the previous same action
#         if current_key in self.past_board:
#             idx = self.past_board.index(current_key)
#             previous_posi = self.past_actions[idx][0]
#             if previous_posi in self.all_move:
#                 moves = self.all_move[previous_posi]
#                 previous_move = self.past_actions[idx][1]
#                 if previous_move in moves:
#                     moves.remove(previous_move)
#                     if len(moves) > 0:
#                         self.all_move[previous_posi] = moves
#                     else:
#                         self.all_move.pop(previous_posi, None)

#         if len(self.all_move) > 0:
#             return True
#         else:
#             return False

#     def __random_action(self):
#         posi_num = len(self.all_move)
#         posi_idx = np.random.randint(posi_num)
#         idx = 0
#         for key in self.all_move:
#             if idx == posi_idx:
#                 posi = key
#                 break
#             idx += 1
#         move_num = len(self.all_move[posi])
#         move_idx = np.random.randint(move_num)
#         idx = 0
#         for moves in self.all_move[posi]:
#             if idx == move_idx:
#                 move = moves
#                 break
#             idx += 1
#         if self.color == 'b':       # rotate move
#             return rotate_action(posi, move)
#         return posi, move

#     def __exploit_action(self):
#         state = torch.from_numpy(copy.deepcopy(self.current_board)).to(self.device)
#         if str(self.device) == 'cuda':
#             state = state.view(1, 1, 10, 9).type(torch.cuda.FloatTensor)
#         else:
#             state = state.view(1, 1, 10, 9).type(torch.FloatTensor)
#         queue = []
#         heapify(queue)
#         for posi in self.all_move:
#             for move in self.all_move[posi]:
#                 next_board = copy.deepcopy(self.current_board)
#                 value = next_board[posi[0]][posi[1]]
#                 next_board[posi[0]][posi[1]] = 0
#                 next_board[posi[0]+move[0]][posi[1]+move[1]] = value
#                 action = torch.from_numpy(next_board).to(self.device)
#                 if str(self.device) == 'cuda':
#                     action = action.view(1, 1, 10, 9).type(torch.cuda.FloatTensor)
#                 else:
#                     action = action.view(1, 1, 10, 9).type(torch.FloatTensor)
#                 win_rate = self.q_star(state, action)
#                 win_rate = win_rate.cpu().detach().numpy()[0][0]
#                 heappush(queue, (-win_rate, (posi, move)))
#         value, action = heappop(queue)
#         if self.color == 'b':       # rotate move
#             return rotate_action(action[0], action[1])
#         return action[0], action[1]
    
#     def ai_action(self):
#         if self.explore_rate == 0:
#             return self.__exploit_action()
#         elif self.explore_rate == 1:
#             return self.__random_action()
#         else:
#             sample = random.random()
#             if sample > self.explore_rate:
#                 return self.__exploit_action()
#             else:
#                 return self.__random_action()

