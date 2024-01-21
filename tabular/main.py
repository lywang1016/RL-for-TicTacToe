import os
import argparse
from tqdm import trange
from game import HumanHumanGame, HumanAIGame, AIHumanGame, AIAIGame

def human_human(total):
    game = HumanHumanGame()
    rwin = 0
    bwin = 0
    t = 0
    for i in range(total):
        winner = game.episode()
        if winner == 'r':
            rwin += 1
        elif winner == 'b':
            bwin += 1
        else:
            t += 1
    print("R win times: " + str(rwin) + "\tR not lose rate: " + str((rwin+t) / total))
    print("B win times: " + str(bwin) + "\tB not lose rate: " + str((bwin+t) / total))
    print("Tie times: " + str(t) + "\t\tTie rate: " + str(t / total))

def human_ai(total, esp):
    game = HumanAIGame(esp)
    rwin = 0
    bwin = 0
    t = 0
    for i in range(total):
        winner = game.episode()
        if winner == 'r':
            rwin += 1
        elif winner == 'b':
            bwin += 1
        else:
            t += 1
    print("R win times: " + str(rwin) + "\tR not lose rate: " + str((rwin+t) / total))
    print("B win times: " + str(bwin) + "\tB not lose rate: " + str((bwin+t) / total))
    print("Tie times: " + str(t) + "\t\tTie rate: " + str(t / total))

def ai_human(total, esp):
    game = AIHumanGame(esp)
    rwin = 0
    bwin = 0
    t = 0
    for i in range(total):
        winner = game.episode()
        if winner == 'r':
            rwin += 1
        elif winner == 'b':
            bwin += 1
        else:
            t += 1
    print("R win times: " + str(rwin) + "\tR not lose rate: " + str((rwin+t) / total))
    print("B win times: " + str(bwin) + "\tB not lose rate: " + str((bwin+t) / total))
    print("Tie times: " + str(t) + "\t\tTie rate: " + str(t / total))

def ai_ai(total, esp):
    if not os.path.exists('q_value'): 
        os.mkdir('q_value')
    game = AIAIGame(esp, if_gui=False)
    rwin = 0
    bwin = 0
    t = 0
    for i in trange(total):
        winner = game.episode()
        if winner == 'r':
            rwin += 1
        elif winner == 'b':
            bwin += 1
        else:
            t += 1
    print("R win times: " + str(rwin) + "\tR not lose rate: " + str((rwin+t) / total))
    print("B win times: " + str(bwin) + "\tB not lose rate: " + str((bwin+t) / total))
    print("Tie times: " + str(t) + "\t\tTie rate: " + str(t / total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_game', type=int, default=1, help='Number of game play')
    parser.add_argument('--type', type=int, default=0, help='type of game')
    parser.add_argument('--eps', type=float, default=0.0, help='AI eps greedy')
    args = parser.parse_args()

    if args.type == 0:
        print("Human vs. Human")
        human_human(args.num_game)
    elif args.type == 1:
        print("Human vs. AI")
        human_ai(args.num_game, args.eps)
    elif args.type == 2:
        print("AI vs. Human")
        ai_human(args.num_game, args.eps)
    else:
        print("AI vs. AI")
        ai_ai(args.num_game, args.eps)