import argparse
from game import AIAIGame, AIHumanGame, HumanAIGame, HumanHumanGame 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_game', type=int, default=1, help='Number of game play')
    parser.add_argument('--type', type=int, default=0, help='type of game')
    args = parser.parse_args()

    if args.type == 0:
        game = HumanHumanGame()
    elif args.type == 1:
        game = HumanAIGame()
    elif args.type == 2:
        game = AIHumanGame()
    else:
        game = AIAIGame()

    rwin = 0
    bwin = 0
    t = 0
    game_num = args.num_game
    for i in range(game_num):
        print("Game " + str(i+1) + '/' + str(game_num))
        winner = game.episode()
        if winner == 'r':
            rwin += 1
        elif winner == 'b':
            bwin += 1
        else:
            t += 1
    print("R win times: " + str(rwin) + "\tR not lose rate: " + str((rwin+t) / game_num))
    print("B win times: " + str(bwin) + "\tB not lose rate: " + str((bwin+t) / game_num))
    print("Tie times: " + str(t) + "\t\tTie rate: " + str(t / game_num))