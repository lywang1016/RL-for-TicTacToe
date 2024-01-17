import math
import torch as T
import numpy as np

class Node:
    def __init__(self, game, args, state, player, parent=None, action_taken=None, prior=0):
        self.game = game
        self.args = args
        self.state = state
        self.player = player  # player should play this node state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = []
        
        self.visit_count = 0
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
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
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - (child.value_sum / child.visit_count)
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, self.player)
                child = Node(self.game, self.args, child_state, self.game.get_opponent(self.player), self, action, prob)
                self.children.append(child)
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  

class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        
    @T.no_grad()
    def search(self, state, player):
        root = Node(self.game, self.args, state, player)
        
        for search in range(self.args['num_searches']):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, -node.player)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                encoded_state = self.game.get_encoded_state(node.state, node.player)
                policy, value = self.model(
                    T.tensor(encoded_state).float().to(self.model.device).unsqueeze(0)
                )
                policy = T.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                value = value.item()

                valid_moves = self.game.get_valid_moves(node.state.copy(), node.player)
                mask = self.game.valid_moves_mask(valid_moves)

                if np.sum(policy*mask) != 0:
                    policy *= mask
                    policy /= np.sum(policy)
                else:
                    policy = mask / np.sum(mask)
                
                node.expand(policy)
                
            node.backpropagate(value)    

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
    
if __name__ == '__main__':
    from network import ResNet
    from tictactoe import TicTacToe

    tictactoe = TicTacToe()
    player = 1

    args = {
        'C': 2,
        'num_searches': 1000
    }

    model = ResNet(tictactoe, 4, 64)
    model.eval()

    mcts = MCTS(tictactoe, args, model)

    state = tictactoe.get_initial_state()

    while True:
        print(state)
        
        if player == 1:
            valid_moves = tictactoe.get_valid_moves(state, player)
            print("valid_moves", valid_moves)
            action = int(input(f"{player}:"))
        else:
            mcts_probs = mcts.search(state.copy(), player)
            action = np.argmax(mcts_probs)
            
        state = tictactoe.get_next_state(state.copy(), action, player)
        
        value, is_terminal = tictactoe.get_value_and_terminated(state, player)
        
        if is_terminal:
            print(state)
            if value == 1:
                print(player, "won")
            else:
                print("draw")
            break
            
        player = tictactoe.get_opponent(player)