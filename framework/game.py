import time
from framework.board import ChessBoard
from framework.display import GUI
# from framework.player import HumanPlayer, AIPlayer
from framework.player import HumanPlayer

class Game():
    def __init__(self, r_type, b_type, if_record=False, if_dataset=False, 
                 if_gui=True, gui_update=0.1, ai_explore_rate=1):
        self.r_type = r_type
        self.b_type = b_type
        self.if_gui = if_gui
        self.gui_update = gui_update
        self.if_record = if_record
        self.if_dataset = if_dataset
        self.red = True
        self.chess_board = ChessBoard()
        if r_type == 'human' and b_type == 'human': # human vs human
            self.if_gui = True
            self.gui = GUI()
            self.r_player = HumanPlayer('r')
            self.b_player = HumanPlayer('b')
        # elif r_type == 'human':                     # human vs AI
        #     self.if_gui = True  
        #     self.ai_explore_rate = ai_explore_rate                  
        #     self.gui = GUI()
        #     self.r_player = HumanPlayer('r')
        #     self.b_player = AIPlayer('b', self.ai_explore_rate)
        # elif b_type == 'human':                     # AI vs human
        #     self.if_gui = True
        #     self.ai_explore_rate = ai_explore_rate                   
        #     self.gui = GUI()
        #     self.r_player = AIPlayer('r', self.ai_explore_rate)
        #     self.b_player = HumanPlayer('b')
        # else:                                       # AI vs AI
        #     if self.if_gui:
        #         self.gui = GUI()
        #     self.ai_explore_rate = ai_explore_rate 
        #     self.r_player = AIPlayer('r', self.ai_explore_rate)
        #     self.b_player = AIPlayer('b', self.ai_explore_rate)
    
    def reset(self):
        self.chess_board.reset_board()
        self.r_player.reset()
        self.b_player.reset()
        self.red = True

    def episode(self):
        if self.r_type == 'human' and self.b_type == 'human':   # human vs human
            self.__human_human_episode()
        # elif self.r_type == 'human' or self.b_type == 'human':  # human vs AI or AI vs human
        #     self.__human_ai_episode()
        # else:                                                   # AI vs AI
        #     self.__ai_ai_episode()

    def __human_human_episode(self):
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
        
        if self.if_record:
            self.chess_board.save_csv()
        if self.if_dataset:
            self.chess_board.fill_dataset()

    # def __human_ai_episode(self):
    #     self.reset()
    #     while not self.chess_board.done:
    #         if self.red:
    #             self.gui.update(self.chess_board.board_states(), 'r')
    #         else:
    #             self.gui.update(self.chess_board.board_states(), 'b')
    #         time.sleep(self.gui_update)
    #         info, position = self.gui.check_event()

    #         if info == 'reset':
    #             print('reset')
    #             break
    #         elif info == 'turn180':
    #             print('turn180')
    #             self.chess_board.rotate_board()
    #             self.red = not self.red
    #             self.r_player.reset()
    #             self.b_player.reset()
    #         elif info == 'tie':
    #             print('mark as tie')
    #             self.chess_board.set_done('t')
    #             break
    #         elif info == 'grid':
    #             if self.r_type == 'human':
    #                 if self.red: #check whos turn
    #                     if self.r_player.stage == 'pick':
    #                         self.r_player.select_piece(position)
    #                     if self.r_player.stage == 'go':
    #                         self.r_player.action_valid(position)
    #                         if self.r_player.move:
    #                             self.chess_board.move_piece(self.r_player.current_piece_posi, self.r_player.move)
    #                             self.red = not self.red
    #             else:
    #                 if not self.red: #check whos turn 
    #                     if self.b_player.stage == 'pick':
    #                         self.b_player.select_piece(position)
    #                     if self.b_player.stage == 'go':
    #                         self.b_player.action_valid(position)
    #                         if self.b_player.move:
    #                             self.chess_board.move_piece(self.b_player.current_piece_posi, self.b_player.move)
    #                             self.red = not self.red
    #         else:
    #             if self.r_type == 'human':
    #                 if self.red:
    #                     self.r_player.update_board(self.chess_board.board_states())
    #                     if not self.r_player.check_moves():
    #                         self.chess_board.set_done('b')
    #                         break
    #                 else:
    #                     self.b_player.update_board(self.chess_board.board_states(), \
    #                                                 self.chess_board.black_history, \
    #                                                 self.chess_board.black_action_history)
    #                     if not self.b_player.check_moves():
    #                         self.chess_board.set_done('r')
    #                         break
    #                     posi, move = self.b_player.ai_action()
    #                     self.chess_board.move_piece(posi, move)
    #                     self.red = not self.red
    #             else:
    #                 if not self.red:
    #                     self.b_player.update_board(self.chess_board.board_states())
    #                     if not self.b_player.check_moves():
    #                         self.chess_board.set_done('r')
    #                         break
    #                 else:
    #                     self.r_player.update_board(self.chess_board.board_states(), \
    #                                                 self.chess_board.red_history, \
    #                                                 self.chess_board.red_action_history)
    #                     if not self.r_player.check_moves():
    #                         self.chess_board.set_done('b')
    #                         break
    #                     posi, move = self.r_player.ai_action()
    #                     self.chess_board.move_piece(posi, move)
    #                     self.red = not self.red
    #     if self.chess_board.win == 'r':
    #         print('Red Win!')
    #     if self.chess_board.win == 'b':
    #         print('Black Win!')
    #     if self.chess_board.win == 't':
    #         print('Tie!')

    #     if self.if_record:
    #         self.chess_board.save_csv()
    #     if self.if_dataset:
    #         self.chess_board.fill_dataset()

    # def __ai_ai_episode(self):
    #     self.reset()
    #     max_step = 1000
    #     step = 0
    #     while not self.chess_board.done:
    #         if step > max_step:
    #             print('Reach maximum step! Mark as tie!')
    #             self.chess_board.set_done('t')
    #             break

    #         if self.if_gui:
    #             if self.red:
    #                 self.gui.update(self.chess_board.board_states(), 'r')
    #             else:
    #                 self.gui.update(self.chess_board.board_states(), 'b')
    #             time.sleep(self.gui_update)
    #             info, position = self.gui.check_event()
    #             if info == 'reset':
    #                 print('reset')
    #                 break
    #             if info == 'turn180':
    #                 print('turn180')
    #                 self.chess_board.rotate_board()
    #                 self.red = not self.red
    #                 self.r_player.reset()
    #                 self.b_player.reset()
    #             if info == 'tie':
    #                 print('mark as tie')
    #                 self.chess_board.set_done('t')
    #                 break

    #         if self.red:
    #             self.r_player.update_board(self.chess_board.board_states(), \
    #                                         self.chess_board.red_history, \
    #                                         self.chess_board.red_action_history)
    #             if not self.r_player.check_moves():
    #                 self.chess_board.set_done('b')
    #                 break
    #             posi, move = self.r_player.ai_action()
    #             self.chess_board.move_piece(posi, move)
    #             self.red = not self.red
    #         else:
    #             self.b_player.update_board(self.chess_board.board_states(), \
    #                                         self.chess_board.black_history, \
    #                                         self.chess_board.black_action_history)
    #             if not self.b_player.check_moves():
    #                 self.chess_board.set_done('r')
    #                 break
    #             posi, move = self.b_player.ai_action()
    #             self.chess_board.move_piece(posi, move)
    #             self.red = not self.red
    #         step += 1


    #     if self.chess_board.win == 'r':
    #         print('Red Win!')
    #     if self.chess_board.win == 'b':
    #         print('Black Win!')
    #     if self.chess_board.win == 't':
    #         print('Tie!')

    #     if self.if_record:
    #         self.chess_board.save_csv()
    #     if self.if_dataset:
    #         self.chess_board.fill_dataset()