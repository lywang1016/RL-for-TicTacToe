import argparse
import time
from board import ChessBoard
from display import GUI
from player_human import HumanPlayer
from player_ai import AIPlayer

class HumanAIGame:
    def __init__(self):
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
                    posi, move = self.b_player.random_action()
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

    game = HumanAIGame()
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