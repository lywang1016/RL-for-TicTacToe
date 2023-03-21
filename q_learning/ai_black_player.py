from os.path import exists
import numpy as np
from player import Player

class AIBlackPlayer(Player):
    def __init__(self):
        self.faction = -1
        self.current_board = None
        self.all_move = []

    def random_action(self):
        posi_num = len(self.all_move)
        posi_idx = np.random.randint(posi_num)
        return self.all_move[posi_idx], self.faction

    