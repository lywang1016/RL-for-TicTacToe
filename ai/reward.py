from framework.constant import piece_values

def two_in_line(input_list):
    if input_list[0] == input_list[1] and input_list[1] == piece_values['b_piece'] and input_list[2] == 0:
        return -1
    if input_list[0] == input_list[2] and input_list[2] == piece_values['b_piece'] and input_list[1] == 0:
        return -1
    if input_list[2] == input_list[1] and input_list[1] == piece_values['b_piece'] and input_list[0] == 0:
        return -1
    if input_list[0] == input_list[1] and input_list[1] == piece_values['r_piece'] and input_list[2] == 0:
        return 1
    if input_list[0] == input_list[2] and input_list[2] == piece_values['r_piece'] and input_list[1] == 0:
        return 1
    if input_list[2] == input_list[1] and input_list[1] == piece_values['r_piece'] and input_list[0] == 0:
        return 1
    return 0

def reward_function(board):
    for i in range(3):
        if board[i][0]==board[i][1] and board[i][1]==board[i][2]:
            if board[i][0]==piece_values['r_piece']:
                return 10
    for i in range(3):
        if board[0][i]==board[1][i] and board[1][i]==board[2][i]:
            if board[0][i]==piece_values['r_piece']:
                return 10
    v1 = board[0][0]
    v2 = board[1][1]
    v3 = board[2][2]
    if v1==v2 and v2==v3:
        if v1==piece_values['r_piece']:
            return 10
    v1 = board[0][2]
    v3 = board[2][0]
    if v1==v2 and v2==v3:
        if v1==piece_values['r_piece']:
            return 10

    reward = 0
    for i in range(3):
        input_list = [board[i][0], board[i][1], board[i][2]]
        val = two_in_line(input_list)
        if val == -1:
            return -10
        if val == 1:
            reward += 1
    for i in range(3):
        input_list = [board[0][i], board[1][i], board[2][i]]
        val = two_in_line(input_list)
        if val == -1:
            return -10
        if val == 1:
            reward += 1
    input_list = [board[0][2], board[1][1], board[2][0]]
    val = two_in_line(input_list)
    if val == -1:
        return -10
    if val == 1:
        reward += 1
    input_list = [board[0][0], board[1][1], board[2][2]]
    val = two_in_line(input_list)
    if val == -1:
        return -10
    if val == 1:
        reward += 1
    return reward