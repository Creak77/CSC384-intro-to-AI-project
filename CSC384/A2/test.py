def max_value(state, turn, alpha, beta, depth):
    if cutoff_test(state) or depth > 8:
        return evaluate(state.board, turn), state
    v = -99999
    best_move = None
    if turn == 'r':
        opp_turn = 'b'
    elif turn == 'b':
        opp_turn = 'r'
    for a in next_states(state, turn):
        pre_v = v
        min_v, temp = min_value(a, turn, alpha, beta, depth + 1)
        v = max(v, min_v)
        if v != pre_v:
            best_move = a
        if v >= beta:
            return v, a
        alpha = max(alpha, v)
    return v, best_move


def min_value(state, turn, alpha, beta, depth):
    if cutoff_test(state) or depth > 8:
        return evaluate(state.board, turn), state
    v = 99999
    best_move = None
    if turn == 'r':
        opp_turn = 'b'
    elif turn == 'b':
        opp_turn = 'r'
    for a in next_states(state, opp_turn):
        pre_v = v
        max_v, temp = max_value(a, turn, alpha, beta, depth + 1)
        v = min(v, max_v)
        if v != pre_v:
            best_move = a
        if v <= alpha:
            return v, a
        beta = min(beta, v)
    return v, best_move


# reuse the cache only when depth is the same or more shallow

def alpha_beta_pruning(state, maxdepth):
    successor = state
    turn = 'r'
    count = 0
    sol_list = []
    sol_list.append(successor)
    while cutoff_test(successor) != True and count < maxdepth:
        _, successor = max_value(successor, turn, alpha, beta, 0)
        successor.display()
        sol_list.append(successor)
        if turn == 'r':
            turn = 'b'
        else:
            turn = 'r'
        count += 1

    return sol_list