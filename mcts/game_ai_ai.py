import argparse
import time
from board import ChessBoard
from display import GUI
from player_ai import AIPlayer

class AIAIGame:
    def __init__(self, if_gui=True):
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
                # posi, move = self.r_player.random_action()
                posi, move = self.r_player.mcts_action()
                self.chess_board.move_piece(posi, move)
                self.red = not self.red
            else:
                self.b_player.update_board(self.chess_board.board_states())
                self.b_player.check_moves()
                # posi, move = self.b_player.random_action()
                posi, move = self.b_player.mcts_action()
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_game', type=int, default=1, help='Number of game play')
    args = parser.parse_args()

    game = AIAIGame()
    rwin = 0
    bwin = 0
    t = 0
    game_num = args.num_game
    for i in range(game_num):
        print("Game " + str(i+1) + '/' + str(game_num))
        winner = game.episode()
        if winner == 'r':
            rwin += 1
        elif winner == 'b':
            bwin += 1
        else:
            t += 1
    print("R win times: " + str(rwin) + "\tR not lose rate: " + str((rwin+t) / game_num))
    print("B win times: " + str(bwin) + "\tB not lose rate: " + str((bwin+t) / game_num))
    print("Tie times: " + str(t) + "\t\tTie rate: " + str(t / game_num))