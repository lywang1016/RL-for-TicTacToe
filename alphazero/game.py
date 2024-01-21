import time
from board import ChessBoard
from display import GUI
from player_human import HumanPlayer
from player_ai import AIPlayer

class AIHumanGame:
    def __init__(self, ai_action_type):
        self.ai_action_type = ai_action_type
        self.gui = GUI()
        self.gui_update = 0.1
        self.red = True
        self.chess_board = ChessBoard()
        self.r_player = AIPlayer('r')
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
                        self.red = not self.red
            else:
                if not self.red:
                    self.b_player.update_board(self.chess_board.board_states())
                    self.b_player.check_moves()
                else:
                    self.r_player.update_board(self.chess_board.board_states())
                    if self.ai_action_type == 0:
                        posi, move = self.r_player.random_action()
                    elif self.ai_action_type == 1:
                        posi, move = self.r_player.mcts_action()
                    elif self.ai_action_type == 2:
                        posi, move = self.r_player.model_action()
                    else:
                        posi, move = self.r_player.alpha_mcts_action()
                    self.chess_board.move_piece(posi, move)
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

class HumanAIGame:
    def __init__(self, ai_action_type):
        self.ai_action_type = ai_action_type
        self.gui = GUI()
        self.gui_update = 0.1
        self.red = True
        self.chess_board = ChessBoard()
        self.r_player = HumanPlayer('r')
        self.b_player = AIPlayer('b')
    
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
                        self.red = not self.red
            else:
                if self.red:
                    self.r_player.update_board(self.chess_board.board_states())
                    self.r_player.check_moves()
                else:
                    self.b_player.update_board(self.chess_board.board_states())
                    if self.ai_action_type == 0:
                        posi, move = self.b_player.random_action()
                    elif self.ai_action_type == 1:
                        posi, move = self.b_player.mcts_action()
                    elif self.ai_action_type == 2:
                        posi, move = self.b_player.model_action()
                    else:
                        posi, move = self.b_player.alpha_mcts_action()
                    self.chess_board.move_piece(posi, move)
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

class AIAIGame:
    def __init__(self, ai_action_type, if_gui=True):
        self.ai_action_type = ai_action_type
        self.if_gui = if_gui
        self.red = True
        self.chess_board = ChessBoard()
        if self.if_gui:
            self.gui = GUI()
            self.gui_update = 1.0
        self.r_player = AIPlayer('r')
        self.b_player = AIPlayer('b')
    
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
                if self.ai_action_type == 0:
                    posi, move = self.r_player.random_action()
                elif self.ai_action_type == 1:
                    posi, move = self.r_player.mcts_action()
                elif self.ai_action_type == 2:
                    posi, move = self.r_player.model_action()
                else:
                    posi, move = self.r_player.alpha_mcts_action()
                self.chess_board.move_piece(posi, move)
                self.red = not self.red
            else:
                self.b_player.update_board(self.chess_board.board_states())
                self.b_player.check_moves()
                if self.ai_action_type == 0:
                    posi, move = self.b_player.random_action()
                elif self.ai_action_type == 1:
                    posi, move = self.b_player.mcts_action()
                elif self.ai_action_type == 2:
                    posi, move = self.b_player.model_action()
                else:
                    posi, move = self.b_player.alpha_mcts_action()
                self.chess_board.move_piece(posi, move)
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

class HumanHumanGame:
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