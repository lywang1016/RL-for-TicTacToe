from framework.game import Game

def main():
    game = Game(r_type='human', b_type='ai', if_record=True, if_dataset=False, ai_explore_rate=0)
    # game = Game(r_type='ai', b_type='human', if_record=True, if_dataset=False, ai_explore_rate=0)
    # game = Game(r_type='ai', b_type='ai', if_record=True, if_dataset=False, if_gui=True, gui_update=1.0, ai_explore_rate=0)

    game.episode()

if __name__ == '__main__':
    main()