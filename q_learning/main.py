from game import HumanHumanGame, HumanAIGame, AIAIGame

def main():
    # game = HumanHumanGame()
    game = HumanAIGame()
    # game = AIAIGame(if_gui=False)

    for i in range(5):
        game.episode()

if __name__ == '__main__':
    main()