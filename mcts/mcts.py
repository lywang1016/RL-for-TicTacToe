import math
import random
import numpy as np

class Node:
    def __init__(self, game, args, color, state, parent=None, action_taken=None):
        self.game = game
        self.args = args
        self.color = color
        if color == 'r':
            self.state = state
        else:
            self.state = game.change_perspective(state, -1)
        self.parent = parent
        self.action_taken = action_taken
        
        self.children = []
        
        self.visit_count = 0
        self.value_sum = 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        for child in self.children:
            if child.visit_count > 0:
                ucb = self.get_ucb(child)
            else:
                ucb = np.inf
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb  
        return best_child
    
    def get_ucb(self, child):
        q_value = child.value_sum / child.visit_count
        q_value = 1 - q_value
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)
    
    def rollout(self):
        value, is_terminal = self.game.get_value_and_terminated(self.state)
        # value = self.game.get_opponent_value(value)
        
        if is_terminal:
            return value
        
        rollout_state = self.state.copy()
        rollout_player = 1
        while True:
            valid_moves = self.game.get_valid_moves(rollout_state)
            action = random.choice(valid_moves)
            rollout_state = self.game.get_next_state(rollout_state, action)
            rollout_state = self.game.change_perspective(rollout_state, -1)
            value, is_terminal = self.game.get_value_and_terminated(rollout_state)
            if is_terminal:
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value)
                return value    
            rollout_player = self.game.get_opponent(rollout_player)
        
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)   
    
    def expand(self):
        actions = self.game.get_valid_moves(self.state)
        for action in actions:
            child_state = self.game.get_next_state(self.state.copy(), action)
            child_state = self.game.change_perspective(child_state, -1)
            if self.color == 'r':
                child_color = 'b'
            else:
                child_color = 'r'
            child = Node(self.game, self.args, child_color, child_state, self, action)
            self.children.append(child)
        return self.children[0]
        
class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        
    def search(self, color, state, step_cnt):
        root = Node(self.game, self.args, color, state, step_cnt)
        
        for search in range(self.args['num_searches']):
            node = root
            
            while len(node.children) > 0:
                node = node.select()

            if node.visit_count == 0:
                value = node.rollout()
            else:
                actions = self.game.get_valid_moves(node.state)
                if len(actions) > 0:
                    node = node.expand()
                value = node.rollout()
            node.backpropagate(value)    
            
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
    