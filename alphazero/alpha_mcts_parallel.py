import math
import torch as T
import numpy as np

class AlphaNode:
    def __init__(self, game, args, state, player, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.player = player  # player should play this node state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = []
        
        self.visit_count = visit_count
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
                child = AlphaNode(self.game, self.args, child_state, self.game.get_opponent(self.player), self, action, prob)
                self.children.append(child)
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  

class AlphaMCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        
    @T.no_grad()
    def search(self, states, player, spGames):
        encoded_states = []
        for i in range(states.shape[0]):
            encoded_state = self.game.get_encoded_state(states[0].copy(), player)
            encoded_states.append(encoded_state)
        encoded_states = np.stack(encoded_states)

        policy, _ = self.model(
            T.tensor(encoded_states, device=self.model.device)
        )
        policy = T.softmax(policy, axis=1).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])
        
        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves(states[i].copy(), player)
            mask = self.game.valid_moves_mask(valid_moves)
            spg_policy *= mask
            spg_policy /= np.sum(spg_policy)

            spg.root = AlphaNode(self.game, self.args, states[i], player, visit_count=1)
            spg.root.expand(spg_policy)
        
        for search in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()
                    
                value, is_terminal = self.game.get_value_and_terminated(node.state, -node.player)
                value = self.game.get_opponent_value(value)

                if is_terminal:
                    node.backpropagate(value)
                else:
                    spg.node = node
            
            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]
            if len(expandable_spGames) > 0:
                encoded_states = []
                for mappingIdx in expandable_spGames:
                    encoded_state = self.game.get_encoded_state(spGames[mappingIdx].node.state, spGames[mappingIdx].node.player)
                    encoded_states.append(encoded_state)
                encoded_states = np.stack(encoded_states)

                policy, value = self.model(
                    T.tensor(encoded_states).float().to(self.model.device)
                )
                policy = T.softmax(policy, axis=1).cpu().numpy()
                value = value.cpu().numpy()

            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy = policy[i]
                spg_value = self.game.get_opponent_value(value[i])
                
                valid_moves = self.game.get_valid_moves(node.state.copy(), node.player)
                mask = self.game.valid_moves_mask(valid_moves)
                spg_policy *= mask
                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)
                node.backpropagate(spg_value)

