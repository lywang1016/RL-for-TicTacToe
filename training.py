import yaml
import torch
from tqdm import tqdm
from os.path import exists
from framework.game import Game
from framework.utils import merge_dataset, h5py_to_dataset, dataset_to_h5py
from torch.utils.data import DataLoader
from ai.dataset import DICT_Dataset
from ai.network import DQN
from ai.loss import MyLoss

# load configuration
with open('ai/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# define game
# game = Game(r_type='ai', b_type='human', if_record=False, if_dataset=True, ai_explore_rate=0.05)
# game = Game(r_type='human', b_type='ai', if_record=False, if_dataset=True, ai_explore_rate=0.05)
game = Game(r_type='ai', b_type='ai', if_record=False, if_dataset=True, if_gui=False, ai_explore_rate=0.3)

# check if use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load network, loss and define optimizer
q_star = DQN().to(device)
criterion = MyLoss().to(device)
optimizer = torch.optim.RMSprop(q_star.parameters(), lr=config['lr_start'])
if exists(config['save_model_path']):
    checkpoint = torch.load(config['save_model_path'])
    q_star.load_state_dict(checkpoint['model_state_dict'])
q_star.train()

# load the history dataset
max_episode_history = config['save_episode_num']
episode_history = []
cur_idx = 0
print('Load most recent ' + str(max_episode_history) + ' episode history...')
for i in tqdm(range(max_episode_history)):
    x_path = 'dataset/x'+str(i)+'.hdf5'
    y_path = 'dataset/y'+str(i)+'.hdf5'
    if exists(x_path) and exists(y_path):
        dataset = h5py_to_dataset(x_path, y_path)
        episode_history.append(dataset)
    else:
        cur_idx = i
        break
if exists('dataset/cur_idx.txt'):
    f = open('dataset/cur_idx.txt','r')
    info = f.read()
    cur_idx = int(info)
    f.close()

# training loop
print('Start training!')
for i in range(config['total_episode_num']):
    print('++++++++++++++++++++++ Episode '+str(i+1)+' of '+str(config['total_episode_num'])+' ++++++++++++++++++++++')
    print('Run an episode')
    game.episode()                                              
    print('Save data')      # save data
    if len(episode_history) < max_episode_history:              
        episode_history.append(game.chess_board.dataset)
    else:
        episode_history[cur_idx] = game.chess_board.dataset
    x_path = 'dataset/x'+str(cur_idx)+'.hdf5'
    y_path = 'dataset/y'+str(cur_idx)+'.hdf5'
    dataset_to_h5py(game.chess_board.dataset, x_path, y_path)
    cur_idx += 1
    if cur_idx == max_episode_history:
        cur_idx = 0
    f = open('dataset/cur_idx.txt', 'w')
    f.write(str(cur_idx))
    f.close()
    print('Build dataset')      # build dataset
    dataset = {}                                                
    for item in episode_history:
        merge_dataset(dataset, item)
    cur_dataset = DICT_Dataset(dataset)
    dataloader = DataLoader(cur_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    print('Train 1 epoch')      # train 1 epoch
    for i, sample in tqdm(enumerate(dataloader), total=len(dataloader), smoothing=0.9):
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
    print('Update model')      # update model
    state = {'model_state_dict': q_star.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    torch.save(state, config['save_model_path'])