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
        self.reset()

class HumanAIGame():
    def __init__(self):
        self.gui = GUI()
        self.gui_update = 0.1
        self.red = True
        self.chess_board = ChessBoard()
        self.r_player = HumanPlayer('r')
        self.b_player = AIBlackPlayer()
    
    def reset(self):
        self.chess_board.reset_board()
        self.r_player.reset()
        self.b_player.reset()
        self.red = True

    def episode(self):
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
                if self.red: # human move
                    posi, value = self.r_player.human_action(position)
                    if value:
                        self.chess_board.move_piece(posi, value)
                        self.b_player.q_update(self.chess_board.board_states(), self.red, self.chess_board.done, self.chess_board.win)
                        self.red = not self.red
            else:
                if self.red:
                    self.r_player.update_board(self.chess_board.board_states())
                    self.r_player.check_moves()
                else:
                    self.b_player.update_board(self.chess_board.board_states())
                    posi, move = self.b_player.eps_greedy_action()
                    self.chess_board.move_piece(posi, move)
                    self.b_player.q_update(self.chess_board.board_states(), self.red, self.chess_board.done, self.chess_board.win)
                    self.red = not self.red
        if self.chess_board.win == 'r':
            print('Red Win!')
        if self.chess_board.win == 'b':
            print('Black Win!')
        if self.chess_board.win == 't':
            print('Tie!')
        self.reset()

# class HumanAIGame():
#     def __init__(self, ai_type):
#         self.gui = GUI()
#         self.gui_update = 0.1
#         self.red = True
#         self.chess_board = ChessBoard()
#         self.ai_type = ai_type
#         if self.ai_type == 'r':
#             self.r_player = AIRedPlayer()
#             self.b_player = HumanPlayer('b')
#         else:
#             self.r_player = HumanPlayer('r')
#             self.b_player = AIBlackPlayer()
    
#     def reset(self):
#         self.chess_board.reset_board()
#         self.r_player.reset()
#         self.b_player.reset()
#         self.red = True

#     def episode(self):
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
#                 if self.ai_type == 'b':
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
#                 if self.ai_type == 'b':
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
#                         posi, move = self.b_player.eps_greedy_action()
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
#                         posi, move = self.r_player.random_action()
#                         self.chess_board.move_piece(posi, move)
#                         self.red = not self.red
#         if self.chess_board.win == 'r':
#             print('Red Win!')
#         if self.chess_board.win == 'b':
#             print('Black Win!')
#         if self.chess_board.win == 't':
#             print('Tie!')

class AIAIGame():
    def __init__(self, if_gui=True):
        self.if_gui = if_gui
        self.red = True
        self.chess_board = ChessBoard()
        if self.if_gui:
            self.gui = GUI()
            self.gui_update = 0.1
        self.r_player = AIRedPlayer()
        self.b_player = AIBlackPlayer()
    
    def reset(self):
        self.chess_board.reset_board()
        self.r_player.reset()
        self.b_player.reset()
        self.red = True

    def episode(self):
        while not self.chess_board.done:
            if self.if_gui:
                if self.red:
                    self.gui.update(self.chess_board.board_states(), 'r')
                else:
                    self.gui.update(self.chess_board.board_states(), 'b')
                time.sleep(self.gui_update)
                info, position = self.gui.check_event()
                if info == 'reset':
                    print('reset')
                    break

            if self.red:
                self.r_player.update_board(self.chess_board.board_states())
                self.r_player.check_moves()
                posi, move = self.r_player.random_action()
                self.chess_board.move_piece(posi, move)
                self.b_player.q_update(self.chess_board.board_states(), self.red, self.chess_board.done, self.chess_board.win)
                self.red = not self.red
            else:
                self.b_player.update_board(self.chess_board.board_states())
                self.b_player.check_moves()
                posi, move = self.b_player.eps_greedy_action()
                self.chess_board.move_piece(posi, move)
                self.b_player.q_update(self.chess_board.board_states(), self.red, self.chess_board.done, self.chess_board.win)
                self.red = not self.red

        if self.chess_board.win == 'r':
            print('Red Win!')
        if self.chess_board.win == 'b':
            print('Black Win!')
        if self.chess_board.win == 't':
            print('Tie!')
        self.reset()