# from framework.display import GUI
# import numpy as np
# import time

# gui = GUI()

# while True:
#     gui.update(np.zeros((3, 3)), 'r')
#     gui.check_event()
#     time.sleep(0.1)

from framework.game import Game

def main():
    game = Game(r_type='human', b_type='human', if_record=False, if_dataset=True, ai_explore_rate=0)
    while True:
        game.episode()

if __name__ == '__main__':
    main()