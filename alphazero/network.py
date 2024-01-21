import torch as T
import torch.nn as nn
import torch.nn.functional as F

T.manual_seed(0)

class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden):
        super().__init__()
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Sigmoid()
        )
        
        self.to(self.device)
        
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
        
        
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
    
if __name__ == '__main__':
    from tictactoe import TicTacToe
    import matplotlib.pyplot as plt
    import numpy as np

    test = TicTacToe()
    state = test.get_initial_state()
    state[0,1] = 1
    encoded_state = test.get_encoded_state(state, player=-1)

    model = ResNet(test, 4, 64)
    model.eval()
    
    tensor_state = T.tensor(encoded_state).float().to(model.device).unsqueeze(0)

    policy, value = model(tensor_state)
    value = value.item()
    print(value)

    policy = T.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

    valid_moves = test.get_valid_moves(state, player=-1)
    print(valid_moves)
    mask = test.valid_moves_mask(valid_moves)

    policy *= mask
    policy /= np.sum(policy)

    # print(state)
    # print(tensor_state)

    plt.bar(range(test.action_size), policy)
    plt.show()