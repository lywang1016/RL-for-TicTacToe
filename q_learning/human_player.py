from player import Player

class HumanPlayer(Player):
    def human_action(self, posi):
        if posi in self.all_move:
            return posi, self.faction
        else:
            return None, None
