from game import HumanHumanGame, HumanAIGame, AIAIGame

def main():
    # game = HumanHumanGame()
    game = HumanAIGame(ai_type='b')
    # game = AIAIGame(if_gui=False)

    for i in range(1):
        game.episode()

if __name__ == '__main__':
    main()