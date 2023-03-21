import time
from board import ChessBoard
from display import GUI
from human_player import HumanPlayer
from ai_red_player import AIRedPlayer
from ai_black_player import AIBlackPlayer

class HumanHumanGame():
    def __init__(self):
        self.gui = GUI()
        self.gui_update = 0.1
        self.red = True
        self.chess_board = ChessBoard()
        self.r_player = HumanPlayer('r')
        self.b_player = HumanPlayer('b')
    
    def reset(self):
        self.chess_board.reset_board()
        self.r_player.reset()
        self.b_player.reset()
        self.red = True

    def episode(self):
        self.reset()
        while not self.chess_board.done:
            if self.red:
                self.gui.update(self.chess_board.board_states(), 'r')
            else:
                self.gui.update(self.chess_board.board_states(), 'b')
            time.sleep(self.gui_update)
            info, position = self.gui.check_event()

            if info == 'reset':
                print('reset')
                break
            elif info == 'grid':
                if self.red: #check whos turn 
                    posi, value = self.r_player.human_action(position)
                    if value:
                        self.chess_board.move_piece(posi, value)
                        self.red = not self.red
                else:   #black move
                    posi, value = self.b_player.human_action(position)
                    if value:
                        self.chess_board.move_piece(posi, value)
                        self.red = not self.red
            else:
                if self.red:
                    self.r_player.update_board(self.chess_board.board_states())
                    self.r_player.check_moves()
                else:
                    self.b_player.update_board(self.chess_board.board_states())
                    self.b_player.check_moves()

        if self.chess_board.win == 'r':
            print('Red Win!')
        if self.chess_board.win == 'b':
            print('Black Win!')
        if self.chess_board.win == 't':
            print('Tie!')


# class Game():
#     def __init__(self, r_type, b_type, if_record=False, if_dataset=False, 
#                  if_gui=True, gui_update=0.1, ai_explore_rate=1):
#         self.r_type = r_type
#         self.b_type = b_type
#         self.if_gui = if_gui
#         self.gui_update = gui_update
#         self.if_record = if_record
#         self.if_dataset = if_dataset
#         self.red = True
#         self.chess_board = ChessBoard()
#         if r_type == 'human' and b_type == 'human': # human vs human
#             self.if_gui = True
#             self.gui = GUI()
#             self.r_player = HumanPlayer('r')
#             self.b_player = HumanPlayer('b')
#         elif r_type == 'human':                     # human vs AI
#             self.if_gui = True  
#             self.ai_explore_rate = ai_explore_rate                  
#             self.gui = GUI()
#             self.r_player = HumanPlayer('r')
#             self.b_player = AIPlayer('b', self.ai_explore_rate)
#         elif b_type == 'human':                     # AI vs human
#             self.if_gui = True
#             self.ai_explore_rate = ai_explore_rate                   
#             self.gui = GUI()
#             self.r_player = AIPlayer('r', self.ai_explore_rate)
#             self.b_player = HumanPlayer('b')
#         else:                                       # AI vs AI
#             if self.if_gui:
#                 self.gui = GUI()
#             self.ai_explore_rate = ai_explore_rate 
#             self.r_player = AIPlayer('r', self.ai_explore_rate)
#             self.b_player = AIPlayer('b', self.ai_explore_rate)
    
#     def reset(self):
#         self.chess_board.reset_board()
#         self.r_player.reset()
#         self.b_player.reset()
#         self.red = True

#     def episode(self):
#         if self.r_type == 'human' and self.b_type == 'human':   # human vs human
#             self.__human_human_episode()
#         elif self.r_type == 'human' or self.b_type == 'human':  # human vs AI or AI vs human
#             self.__human_ai_episode()
#         else:                                                   # AI vs AI
#             self.__ai_ai_episode()

#     def __human_human_episode(self):
#         self.reset()
#         while not self.chess_board.done:
#             if self.red:
#                 self.gui.update(self.chess_board.board_states(), 'r')
#             else:
#                 self.gui.update(self.chess_board.board_states(), 'b')
#             time.sleep(self.gui_update)
#             info, position = self.gui.check_event()

#             if info == 'reset':
#                 print('reset')
#                 break
#             elif info == 'grid':
#                 if self.red: #check whos turn 
#                     posi, value = self.r_player.human_action(position)
#                     if value:
#                         self.chess_board.move_piece(posi, value)
#                         self.red = not self.red
#                 else:   #black move
#                     posi, value = self.b_player.human_action(position)
#                     if value:
#                         self.chess_board.move_piece(posi, value)
#                         self.red = not self.red
#             else:
#                 if self.red:
#                     self.r_player.update_board(self.chess_board.board_states())
#                     self.r_player.check_moves()
#                 else:
#                     self.b_player.update_board(self.chess_board.board_states())
#                     self.b_player.check_moves()

#         if self.chess_board.win == 'r':
#             print('Red Win!')
#         if self.chess_board.win == 'b':
#             print('Black Win!')
#         if self.chess_board.win == 't':
#             print('Tie!')
        
#         if self.if_record:
#             self.chess_board.save_csv()
#         if self.if_dataset:
#             self.chess_board.fill_dataset()

#     def __human_ai_episode(self):
#         self.reset()
#         while not self.chess_board.done:
#             if self.red:
#                 self.gui.update(self.chess_board.board_states(), 'r')
#             else:
#                 self.gui.update(self.chess_board.board_states(), 'b')
#             time.sleep(self.gui_update)
#             info, position = self.gui.check_event()

#             if info == 'reset':
#                 print('reset')
#                 break
#             elif info == 'grid':
#                 if self.r_type == 'human':
#                     if self.red: #check whos turn
#                         posi, value = self.r_player.human_action(position)
#                         if value:
#                             self.chess_board.move_piece(posi, value)
#                             self.red = not self.red
#                 else:
#                     if not self.red: #check whos turn 
#                         posi, value = self.b_player.human_action(position)
#                         if value:
#                             self.chess_board.move_piece(posi, value)
#                             self.red = not self.red
#             else:
#                 if self.r_type == 'human':
#                     if self.red:
#                         self.r_player.update_board(self.chess_board.board_states())
#                         if not self.r_player.check_moves():
#                             self.chess_board.set_done('b')
#                             break
#                     else:
#                         self.b_player.update_board(self.chess_board.board_states())
#                         if not self.b_player.check_moves():
#                             self.chess_board.set_done('r')
#                             break
#                         posi, move = self.b_player.ai_action()
#                         self.chess_board.move_piece(posi, move)
#                         self.red = not self.red
#                 else:
#                     if not self.red:
#                         self.b_player.update_board(self.chess_board.board_states())
#                         if not self.b_player.check_moves():
#                             self.chess_board.set_done('r')
#                             break
#                     else:
#                         self.r_player.update_board(self.chess_board.board_states())
#                         if not self.r_player.check_moves():
#                             self.chess_board.set_done('b')
#                             break
#                         posi, move = self.r_player.ai_action()
#                         self.chess_board.move_piece(posi, move)
#                         self.red = not self.red
#         if self.chess_board.win == 'r':
#             print('Red Win!')
#         if self.chess_board.win == 'b':
#             print('Black Win!')
#         if self.chess_board.win == 't':
#             print('Tie!')

#         if self.if_record:
#             self.chess_board.save_csv()
#         if self.if_dataset:
#             self.chess_board.fill_dataset()

#     def __ai_ai_episode(self):
#         self.reset()
#         while not self.chess_board.done:
#             if self.if_gui:
#                 if self.red:
#                     self.gui.update(self.chess_board.board_states(), 'r')
#                 else:
#                     self.gui.update(self.chess_board.board_states(), 'b')
#                 time.sleep(self.gui_update)
#                 info, position = self.gui.check_event()
#                 if info == 'reset':
#                     print('reset')
#                     break

#             if self.red:
#                 self.r_player.update_board(self.chess_board.board_states())
#                 if not self.r_player.check_moves():
#                     self.chess_board.set_done('b')
#                     break
#                 posi, move = self.r_player.ai_action()
#                 self.chess_board.move_piece(posi, move)
#                 self.red = not self.red
#             else:
#                 self.b_player.update_board(self.chess_board.board_states())
#                 if not self.b_player.check_moves():
#                     self.chess_board.set_done('r')
#                     break
#                 posi, move = self.b_player.ai_action()
#                 self.chess_board.move_piece(posi, move)
#                 self.red = not self.red

#         if self.chess_board.win == 'r':
#             print('Red Win!')
#         if self.chess_board.win == 'b':
#             print('Black Win!')
#         if self.chess_board.win == 't':
#             print('Tie!')

#         if self.if_record:
#             self.chess_board.save_csv()
#         if self.if_dataset:
#             self.chess_board.fill_dataset()