import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, h=3, w=3, outputs=1):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(12, 32, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        
        fc_dim_input = 2048

        self.fc_1 = nn.Linear(fc_dim_input, 2*fc_dim_input)
        self.bn_1 = nn.BatchNorm1d(2*fc_dim_input)
        self.fc_2 = nn.Linear(2*fc_dim_input, fc_dim_input)
        self.bn_2 = nn.BatchNorm1d(fc_dim_input)
        self.fc_3 = nn.Linear(fc_dim_input, 32*outputs)
        self.bn_3 = nn.BatchNorm1d(32*outputs)

        self.fc_out = nn.Linear(32*outputs, outputs)
        self.bn_out = nn.BatchNorm1d(outputs)

        self.relu = nn.LeakyReLU()

    def forward(self, state, action):
        batch_size = state.shape[0]
        x_state = state.view(batch_size, 6, 3, 3)
        x_action = action.view(batch_size, 6, 3, 3)
        x = torch.cat((x_state, x_action), 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn_1(self.fc_1(x.view(x.size(0), -1))))
        x = self.relu(self.bn_2(self.fc_2(x)))
        x = self.relu(self.bn_3(self.fc_3(x)))
        y = self.bn_out(self.fc_out(x))
        return y