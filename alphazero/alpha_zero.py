import random
import torch as T
import torch.nn.functional as F
import numpy as np
from tqdm import trange
from alpha_mcts import MCTS

class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
        
    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()
        
        while True:
            action_probs = self.mcts.search(state.copy(), player)
            memory.append((state.copy(), action_probs, player))
            action = np.random.choice(self.game.action_size, p=action_probs)
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
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])] 
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
                T.save(self.model.state_dict(), self.args['model_path']+f"model_{iteration+1}.pth")
                T.save(self.optimizer.state_dict(), self.args['model_path']+f"optimizer_{iteration+1}.pth")

if __name__ == '__main__':
    import os
    import numpy as np
    from tictactoe import TicTacToe
    from network import ResNet

    tictactoe = TicTacToe()

    model = ResNet(tictactoe, 4, 64)

    optimizer = T.optim.Adam(model.parameters(), lr=0.001)

    if not os.path.exists('model'): 
        os.mkdir('model')

    args = {
        'C': 2,
        'num_searches': 300,
        'num_iterations': 2,
        'num_selfPlay_iterations': 500,
        'num_epochs': 8,
        'batch_size': 64,
        'save_every': 1,
        'model_path': 'model/'
    }

    alphaZero = AlphaZero(model, optimizer, tictactoe, args)
    alphaZero.learn()
