import math
import numpy as np

class Node:
    def __init__(self, game, args, state, player, parent=None, action_taken=None):
        self.game = game
        self.args = args
        self.state = state
        self.player = player  # player should play this node state
        self.parent = parent
        self.action_taken = action_taken
        
        self.children = []
        self.expandable_moves = game.valid_moves_mask(game.get_valid_moves(state.copy(), player))
        
        self.visit_count = 0
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        q_value = 1 - (child.value_sum / child.visit_count)
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)
    
    def expand(self):
        action = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[action] = 0
        
        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, action, self.player)
        
        child = Node(self.game, self.args, child_state, self.game.get_opponent(self.player), self, action)
        self.children.append(child)
        return child
    
    def simulate(self):
        value, is_terminal = self.game.get_value_and_terminated(self.state, self.parent.player)
        value = self.game.get_opponent_value(value)
        if is_terminal:
            return value
        
        rollout_state = self.state.copy()
        rollout_player = self.player
        while True:
            valid_moves = self.game.valid_moves_mask(self.game.get_valid_moves(rollout_state.copy(), rollout_player))
            action = np.random.choice(np.where(valid_moves == 1)[0])
            rollout_state = self.game.get_next_state(rollout_state.copy(), action, rollout_player)
            value, is_terminal = self.game.get_value_and_terminated(rollout_state, rollout_player)
            if is_terminal:
                if rollout_player != self.player:
                    value = self.game.get_opponent_value(value)
                return value 
            
            rollout_player = self.game.get_opponent(rollout_player)
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  

class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        
    def search(self, state, player):
        root = Node(self.game, self.args, state, player)
        
        for search in range(self.args['num_searches']):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, -node.player)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                node = node.expand()
                value = node.simulate()
                
            node.backpropagate(value)    

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tictactoe import TicTacToe

    tictactoe = TicTacToe()
    args = {
        'C': 2,
        'num_searches': 10000
    }
    mcts = MCTS(tictactoe, args)
    state = tictactoe.get_initial_state()
    state[0,0] = -1
    state[0,1] = 1
    state[0,2] = 1
    state[1,0] = 1
    state[1,1] = -1
    state[2,0] = -1
    # state[2,2] = 1
    mcts_probs = mcts.search(state, player=1)

    plt.bar(range(tictactoe.action_size), mcts_probs)
    plt.show()