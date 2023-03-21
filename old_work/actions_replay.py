import csv
import time
from framework.board import ChessBoard
from framework.display import GUI

def actions_replay(csv_path):
    chess_board = ChessBoard()
    gui_update = 1.0
    gui = GUI()
    red = True
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            if red:
                gui.update(chess_board.board_states(), 'r')
            else:
                gui.update(chess_board.board_states(), 'b')
            time.sleep(gui_update)
            gui.check_event()

            posi = (int(row[0][0]), int(row[0][2]))
            if row[0][4] == '-':
                move = -int(row[0][5])
            else:
                move = int(row[0][4])
            
            chess_board.move_piece(posi, move)
            red = not red

    while True:
        if red:
            gui.update(chess_board.board_states(), 'r')
        else:
            gui.update(chess_board.board_states(), 'b')
        time.sleep(gui_update)
        gui.check_event()

if __name__ == '__main__':
    actions_replay('game_record/2022_6_4-2_1_20_881489.csv')