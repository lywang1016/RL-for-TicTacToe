# RL-for-TicTacToe
This is one of my personal project, aiming to learn Reinforcemnet learning. In this project, there are two implemented methods: Tabular RL and AlphaZero.

## Tabular Reinforcement Learning
It support 3 algorithms: SARSA(on-policy), Q-learning(off-policy), and Double-Q-learning(off-policy).

To select the algorithm, you first go to folder tabular, and open file of 'config.yaml'. In parameter 'learn_method_r' and 'learn_method_b', you can put sarsa, q_learning, or double_q_learning. Please notice that player 'r'(play firstly) and player 'b'(play secondly) can use different algorithms.

#### To train model, you need to run:
```
python main.py --num_game 80000 --type 3 --eps 0.05
```
Where the argument 'type' = 3 means AI player vs. AI player. and the argument 'eps' is the rate of choose action randomly.

#### To test your model, you can run:
```
python main.py --num_game 200 --type 3 --eps 0.0
```
In this case you don't want random action, so we set 'eps' = 0. You will see all game results are Tie. No player can win since no player play wrong.

#### To play with your model, you can run:
```
python main.py --num_game 1 --type 1 --eps 0.0
```
or
```
python main.py --num_game 1 --type 2 --eps 0.0
```

You will see a GUI and you can click on the chess board to play. 'type' = 1 you will play first, and 'type' = 2 AI will play first.

#### To play with another human, you can run:
```
python main.py --num_game 1 --type 0
```

## AlphaZero
You first go to folder alphazero. Again, all parameters are in file 'config.yaml'.

The AI player have 3 algorithms. First, the AI can make decision only based on MCTS. To get good results, the MCTS should go deep. Second, if you have a trained a model, the AI can make decision just use model prediction. Third, with the model, the AI can use model prediction do a few MCTS to get best decision (No need to go deep). 

#### To train model, you can run:
```
python alpha_zero_parallel.py
```
for fast training. You can also run:
```
python alpha_zero.py
```
if you have a lot of time.

#### To test your model, you can run:
```
python test.py --num_game 200 --type 3 --ai_action_type 3
```
The argument 'ai_action_type' support different algorithms. '0' for random action, '1' for MCTS only, '2' for use model only, '3' for use model to do few MCTS. 

#### To play with your model, you can run:
```
python test.py --num_game 1 --type 1 --ai_action_type 0/1/2/3
```
or
```
python test.py --num_game 1 --type 2 --ai_action_type 0/1/2/3
```

Again, you will see a GUI and you can click on the chess board to play. 'type' = 1 you will play first, and 'type' = 2 ai will play first.

#### Again, to play with another human, you can run:
```
python test.py --num_game 1 --type 0
```