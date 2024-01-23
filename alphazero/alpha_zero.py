import random
import torch as T
import torch.nn.functional as F
import numpy as np
from tqdm import trange
from alpha_mcts import AlphaMCTS

class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = AlphaMCTS(game, args, model)
        
    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()
        
        while True:
            action_probs = self.mcts.search(state.copy(), player)
            memory.append((state.copy(), action_probs, player))
            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(self.game.action_size, p=temperature_action_probs)
            state = self.game.get_next_state(state.copy(), action, player)

            value, is_terminal = self.game.get_value_and_terminated(state, player)
            if is_terminal:
                returnMemory = []
                for hist_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value
                    returnMemory.append((
                        self.game.get_encoded_state(hist_state, hist_player),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory
            
            player = self.game.get_opponent(player)
                
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
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()
                
            print("\tTraining...")
            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)
            
            if (iteration+1)%self.args['save_every'] == 0:
                T.save(self.model.state_dict(), self.args['model_full_path'])
                # T.save(self.model.state_dict(), self.args['model_path']+"model.pth")
                # T.save(self.optimizer.state_dict(), self.args['model_path']+"optimizer.pth")
                # T.save(self.model.state_dict(), self.args['model_path']+f"model_{iteration+1}.pth")
                # T.save(self.optimizer.state_dict(), self.args['model_path']+f"optimizer_{iteration+1}.pth")

if __name__ == '__main__':
    import os
    import yaml
    import numpy as np
    from tictactoe import TicTacToe
    from network import ResNet

    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tictactoe = TicTacToe()

    model = ResNet(tictactoe, config['num_resBlocks'], config['num_hidden'])
    # model.load_state_dict(T.load(self.config['model_full_path']))

    optimizer = T.optim.Adam(model.parameters(), lr=config['learn_rate'], weight_decay=config['weight_decay'])

    if not os.path.exists('model'): 
        os.mkdir('model')

    args = {
        'C': config['C'],
        'num_searches': config['num_Alpha_MCTS_training_searches'],
        'num_iterations': config['num_iterations'],
        'num_selfPlay_iterations': config['num_selfPlay_iterations'],
        'num_epochs': config['num_epochs'],
        'batch_size': config['batch_size'],
        'temperature': config['temperature'],
        'dirichlet_epsilon': config['dirichlet_epsilon'],
        'dirichlet_alpha': config['dirichlet_alpha'],
        'save_every': config['save_every'],
        'model_full_path': config['model_full_path']
    }

    alphaZero = AlphaZero(model, optimizer, tictactoe, args)
    alphaZero.learn()
