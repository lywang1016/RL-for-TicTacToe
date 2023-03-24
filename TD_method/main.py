from tqdm import trange
from game import HumanHumanGame, HumanAIGame, AIHumanGame, AIAIGame

def human_human():
    game = HumanHumanGame()
    rwin = 0
    bwin = 0
    t = 0
    total = 5
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

def human_ai():
    game = HumanAIGame()
    rwin = 0
    bwin = 0
    t = 0
    total = 5
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

def ai_human():
    game = AIHumanGame()
    rwin = 0
    bwin = 0
    t = 0
    total = 5
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

def ai_ai():
    game = AIAIGame(if_gui=False)
    rwin = 0
    bwin = 0
    t = 0
    total = 20000
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
    # human_human()
    # ai_human()
    # human_ai()
    ai_ai()