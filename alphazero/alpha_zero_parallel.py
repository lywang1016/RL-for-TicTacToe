import random
import torch as T
import torch.nn.functional as F
import numpy as np
from tqdm import trange
from alpha_mcts_parallel import AlphaMCTSParallel
from utils import softmax

class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = AlphaMCTSParallel(game, args, model)
        
    def selfPlay(self):
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]
        
        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            self.mcts.search(states.copy(), player, spGames)

            for i in range(len(spGames))[::-1]:
                spg = spGames[i]
                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                temperature_action_probs = softmax(temperature_action_probs)
                action = np.random.choice(self.game.action_size, p=temperature_action_probs)
                spg.state = self.game.get_next_state(spg.state.copy(), action, player)

                value, is_terminal = self.game.get_value_and_terminated(spg.state, player)

                if is_terminal:
                    for hist_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value
                        return_memory.append((
                            self.game.get_encoded_state(hist_state, hist_player),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del spGames[i]
            
            player = self.game.get_opponent(player)
        return return_memory
                
    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            # sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])] 
            sample = memory[batchIdx:batchIdx+self.args['batch_size']]
            state, policy_targets, value_targets = zip(*sample)
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = T.tensor(state).float().to(self.model.device)
            policy_targets = T.tensor(policy_targets, dtype=T.float32).to(self.model.device)
            value_targets = T.tensor(value_targets, dtype=T.float32).to(self.model.device)
            
            out_policy, out_value = self.model(state)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step() 
    
    def learn(self):
        for iteration in range(self.args['num_iterations']):
            print('---------------------- Iteration ' + str(iteration+1) + '/' + str(self.args['num_iterations']) + ' ----------------------')
            memory = []
            
            print("\tSelf play...")
            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                memory += self.selfPlay()
                
            print("\tTraining...")
            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)
            
            if (iteration+1)%self.args['save_every'] == 0:
                T.save(self.model.state_dict(), self.args['model_path']+"model.pth")
                # T.save(self.optimizer.state_dict(), self.args['model_path']+"optimizer.pth")
                # T.save(self.model.state_dict(), self.args['model_path']+f"model_{iteration+1}.pth")
                # T.save(self.optimizer.state_dict(), self.args['model_path']+f"optimizer_{iteration+1}.pth")

class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None

if __name__ == '__main__':
    import os
    import numpy as np
    from tictactoe import TicTacToe
    from network import ResNet

    tictactoe = TicTacToe()

    model = ResNet(tictactoe, 4, 64)
    # model.load_state_dict(T.load('model/model.pth'))

    optimizer = T.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    if not os.path.exists('model'): 
        os.mkdir('model')

    args = {
        'C': 2,
        'num_searches': 300,
        'num_iterations': 10,
        'num_selfPlay_iterations': 600,
        'num_parallel_games': 200,
        'num_epochs': 8,
        'batch_size': 64,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3,
        'save_every': 1,
        'model_path': 'model/'
    }

    alphaZero = AlphaZeroParallel(model, optimizer, tictactoe, args)
    alphaZero.learn()
