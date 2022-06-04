import yaml
from os.path import exists
import math
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ai.dataset import HDF5_Dataset
from ai.network import DQN
from ai.loss import MyLoss

# check if use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load configuration
with open('ai/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# load dataset
dataset = HDF5_Dataset()
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)

# load network, loss and define optimizer
q_star = DQN().to(device)
criterion = MyLoss().to(device)
optimizer = torch.optim.RMSprop(q_star.parameters()) # optimizer = torch.optim.Adam(q_star.parameters())
if exists(config['save_model_path']):
    checkpoint = torch.load(config['save_model_path'])
    q_star.load_state_dict(checkpoint['model_state_dict'])
q_star.train()

# Train
loss_history = []
for epoch in range(config['epoch']):
    print('---------------------- Eopch '+str(epoch+1)+' ----------------------')
    lr = config['lr_end'] + (config['lr_start'] - config['lr_end']) * \
        math.exp(-1. * epoch / config['lr_decay'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    num_batches = len(dataloader)
    loss_sum = 0
    for i, sample in tqdm(enumerate(dataloader), total=num_batches, smoothing=0.9):
        board = sample['board'].float().to(device)
        action = sample['action'].float().to(device)
        win_rate = sample['win_rate'].float().to(device).view(config['batch_size'], 1)
        predict_win_rate = q_star(board, action)
        loss = criterion(win_rate, predict_win_rate)
        optimizer.zero_grad()
        loss.backward()
        for param in q_star.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        loss_sum += loss
    mean_loss = float(loss_sum) / num_batches
    print('Eopch '+str(epoch)+' mean loss is: '+str(mean_loss))
    loss_history.append(mean_loss)
    if (epoch+1) % config['save_every'] == 0:
        state = {'model_state_dict': q_star.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
        torch.save(state, config['save_model_path'])

# plot loss
epoch = np.linspace(start=1, stop=len(loss_history), num=len(loss_history))
plt.figure()
plt.plot(epoch, loss_history, linewidth=2.0)
plt.xlabel('Epoch')
plt.ylabel('Mean Loss')
plt.title('Loss Change')
plt.show()