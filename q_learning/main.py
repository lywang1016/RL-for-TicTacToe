from game import HumanHumanGame

def main():
    game = HumanHumanGame()

    for i in range(5):
        game.episode()

if __name__ == '__main__':
    main()