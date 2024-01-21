import time
from board import ChessBoard
from display import GUI
from human_player import HumanPlayer
from ai_player import AIPlayer

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

        winner = self.chess_board.win
        if self.chess_board.win == 'r':
            print('Red Win!')
        if self.chess_board.win == 'b':
            print('Black Win!')
        if self.chess_board.win == 't':
            print('Tie!')
        self.reset()
        return winner

class HumanAIGame():
    def __init__(self, esp):
        self.gui = GUI()
        self.gui_update = 0.1
        self.red = True
        self.chess_board = ChessBoard()
        self.r_player = HumanPlayer('r')
        self.b_player = AIPlayer('b', esp)
    
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
        
        winner = self.chess_board.win
        if self.chess_board.win == 'r':
            print('Red Win!')
        if self.chess_board.win == 'b':
            print('Black Win!')
        if self.chess_board.win == 't':
            print('Tie!')
        self.reset()
        return winner

class AIHumanGame():
    def __init__(self, esp):
        self.gui = GUI()
        self.gui_update = 0.1
        self.red = True
        self.chess_board = ChessBoard()
        self.r_player = AIPlayer('r', esp)
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
                if not self.red: # human move
                    posi, value = self.b_player.human_action(position)
                    if value:
                        self.chess_board.move_piece(posi, value)
                        self.r_player.q_update(self.chess_board.board_states(), self.red, self.chess_board.done, self.chess_board.win)
                        self.red = not self.red
            else:
                if not self.red:
                    self.b_player.update_board(self.chess_board.board_states())
                    self.b_player.check_moves()
                else:
                    self.r_player.update_board(self.chess_board.board_states())
                    posi, move = self.r_player.eps_greedy_action()
                    self.chess_board.move_piece(posi, move)
                    self.r_player.q_update(self.chess_board.board_states(), self.red, self.chess_board.done, self.chess_board.win)
                    self.red = not self.red
        
        winner = self.chess_board.win
        if self.chess_board.win == 'r':
            print('Red Win!')
        if self.chess_board.win == 'b':
            print('Black Win!')
        if self.chess_board.win == 't':
            print('Tie!')
        self.reset()
        return winner

class AIAIGame():
    def __init__(self, esp, if_gui=True):
        self.if_gui = if_gui
        self.red = True
        self.chess_board = ChessBoard()
        if self.if_gui:
            self.gui = GUI()
            self.gui_update = 0.1
        self.r_player = AIPlayer('r', esp)
        self.b_player = AIPlayer('b', esp)
    
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
                posi, move = self.r_player.eps_greedy_action()
                self.chess_board.move_piece(posi, move)
                self.r_player.q_update(self.chess_board.board_states(), self.red, self.chess_board.done, self.chess_board.win)
                self.b_player.q_update(self.chess_board.board_states(), self.red, self.chess_board.done, self.chess_board.win)
                self.red = not self.red
            else:
                self.b_player.update_board(self.chess_board.board_states())
                self.b_player.check_moves()
                posi, move = self.b_player.eps_greedy_action()
                self.chess_board.move_piece(posi, move)
                self.r_player.q_update(self.chess_board.board_states(), self.red, self.chess_board.done, self.chess_board.win)
                self.b_player.q_update(self.chess_board.board_states(), self.red, self.chess_board.done, self.chess_board.win)
                self.red = not self.red

        winner = self.chess_board.win
        # if self.chess_board.win == 'r':
        #     print('Red Win!')
        # if self.chess_board.win == 'b':
        #     print('Black Win!')
        # if self.chess_board.win == 't':
        #     print('Tie!')
        self.reset()
        return winner