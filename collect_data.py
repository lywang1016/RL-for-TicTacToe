from framework.game import Game
from framework.utils import merge_dataset, h5py_to_dataset, h5py_add_dataset

def main():
    dataset = h5py_to_dataset('dataset/x.hdf5', 'dataset/y.hdf5')

    # game = Game(r_type='human', b_type='human', if_record=False, if_dataset=True, ai_explore_rate=0)
    game = Game(r_type='ai', b_type='ai', if_record=False, if_dataset=True, if_gui=False, gui_update=1, ai_explore_rate=0.05)
    # game = Game(r_type='human', b_type='ai', if_record=False, if_dataset=True, ai_explore_rate=0.05)
    # game = Game(r_type='ai', b_type='human', if_record=False, if_dataset=True, ai_explore_rate=0.05)

    for i in range(5):
        game.episode()
        merge_dataset(dataset, game.chess_board.dataset)

    h5py_add_dataset('dataset/x.hdf5', 'dataset/y.hdf5', dataset)

if __name__ == '__main__':
    main()