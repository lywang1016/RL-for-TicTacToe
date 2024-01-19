import copy

class Player():
    def __init__(self, color):
        self.faction = 1
        if color == 'b':
            self.faction = -1
        self.current_board = None
        self.all_move = []

    def reset(self):
        self.current_board = None
        self.all_move = []

    def check_moves(self):
        self.all_move = []
        for i in range(3):
            for j in range(3):
                if self.current_board[i][j] == 0:
                    self.all_move.append((i,j))
        return len(self.all_move)
    
    def update_board(self, board):
        self.current_board = copy.deepcopy(board)
