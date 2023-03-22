from game import HumanHumanGame, HumanAIGame, AIAIGame

def main():
    # game = HumanHumanGame()
    game = HumanAIGame()
    # game = AIAIGame(if_gui=False)

    for i in range(20):
        game.episode()

if __name__ == '__main__':
    main()