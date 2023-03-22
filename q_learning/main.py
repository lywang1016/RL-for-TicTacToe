from game import HumanHumanGame, HumanAIGame, AIAIGame

def main():
    # game = HumanHumanGame()
    # game = HumanAIGame()
    game = AIAIGame(if_gui=False)

    rwin = 0
    bwin = 0
    t = 0
    total = 2000
    for i in range(total):
        winner = game.episode()
        if winner == 'r':
            rwin += 1
        elif winner == 'b':
            bwin += 1
        else:
            t += 1
    print("R win times: " + str(rwin))
    print("B win times: " + str(bwin))
    print("Tie times: " + str(t))
    print("B not lose rate: " + str((bwin+t) / total))

if __name__ == '__main__':
    main()