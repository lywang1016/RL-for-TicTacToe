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
    if input_list[0] == input_list[1] and input_list[1] == piece_values['b_piece'] and input_list[2] == piece_values['r_piece']:
        return 2
    if input_list[0] == input_list[2] and input_list[2] == piece_values['b_piece'] and input_list[1] == piece_values['r_piece']:
        return 2
    if input_list[2] == input_list[1] and input_list[1] == piece_values['b_piece'] and input_list[0] == piece_values['r_piece']:
        return 2
    return 0

def get_3(action):
    for i in range(3):
        if action[i][0]==action[i][1] and action[i][1]==action[i][2]:
            if action[i][0]==piece_values['r_piece']:
                return True
    for i in range(3):
        if action[0][i]==action[1][i] and action[1][i]==action[2][i]:
            if action[0][i]==piece_values['r_piece']:
                return True
    v1 = action[0][0]
    v2 = action[1][1]
    v3 = action[2][2]
    if v1==v2 and v2==v3:
        if v1==piece_values['r_piece']:
            return True
    v1 = action[0][2]
    v3 = action[2][0]
    if v1==v2 and v2==v3:
        if v1==piece_values['r_piece']:
            return True
    return False

def leave_2(action):
    for i in range(3):
        input_list = [action[i][0], action[i][1], action[i][2]]
        val = two_in_line(input_list)
        if val == -1:
            return True
    for i in range(3):
        input_list = [action[0][i], action[1][i], action[2][i]]
        val = two_in_line(input_list)
        if val == -1:
            return True
    input_list = [action[0][2], action[1][1], action[2][0]]
    val = two_in_line(input_list)
    if val == -1:
        return True
    input_list = [action[0][0], action[1][1], action[2][2]]
    val = two_in_line(input_list)
    if val == -1:
        return True
    return False

def add_2(board, action):
    add_num = 0
    for i in range(3):
        input_list0 = [board[i][0], board[i][1], board[i][2]]
        val0 = two_in_line(input_list0)
        input_list = [action[i][0], action[i][1], action[i][2]]
        val = two_in_line(input_list)
        if val0 != 1 and val == 1:
            add_num += 1
    for i in range(3):
        input_list0 = [board[0][i], board[1][i], board[2][i]]
        val0 = two_in_line(input_list0)
        input_list = [action[0][i], action[1][i], action[2][i]]
        val = two_in_line(input_list)
        if val0 != 1 and val == 1:
            add_num += 1
    input_list0 = [board[0][2], board[1][1], board[2][0]]
    val0 = two_in_line(input_list0)
    input_list = [action[0][2], action[1][1], action[2][0]]
    val = two_in_line(input_list)
    if val0 != 1 and val == 1:
        add_num += 1
    input_list0 = [board[0][0], board[1][1], board[2][2]]
    val0 = two_in_line(input_list0)
    input_list = [action[0][0], action[1][1], action[2][2]]
    val = two_in_line(input_list)
    if val0 != 1 and val == 1:
        add_num += 1
    return add_num

def block_2(board, action):
    add_num = 0
    for i in range(3):
        input_list0 = [board[i][0], board[i][1], board[i][2]]
        val0 = two_in_line(input_list0)
        input_list = [action[i][0], action[i][1], action[i][2]]
        val = two_in_line(input_list)
        if val0 == -1 and val == 2:
            add_num += 1
    for i in range(3):
        input_list0 = [board[0][i], board[1][i], board[2][i]]
        val0 = two_in_line(input_list0)
        input_list = [action[0][i], action[1][i], action[2][i]]
        val = two_in_line(input_list)
        if val0 == -1 and val == 2:
            add_num += 1
    input_list0 = [board[0][2], board[1][1], board[2][0]]
    val0 = two_in_line(input_list0)
    input_list = [action[0][2], action[1][1], action[2][0]]
    val = two_in_line(input_list)
    if val0 == -1 and val == 2:
        add_num += 1
    input_list0 = [board[0][0], board[1][1], board[2][2]]
    val0 = two_in_line(input_list0)
    input_list = [action[0][0], action[1][1], action[2][2]]
    val = two_in_line(input_list)
    if val0 == -1 and val == 2:
        add_num += 1
    return add_num

def reward_function(board, action):
    if get_3(action):
        return 10
    if leave_2(action):
        return -10
    reward = 0
    reward += 1*add_2(board, action)
    reward += 5*block_2(board, action)
    return reward
