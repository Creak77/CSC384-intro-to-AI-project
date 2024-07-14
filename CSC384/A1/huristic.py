def heuristic(state):
    for y, length in enumerate(state.board.grid):
        for x, value in enumerate(length):
            if value == char_goal:
                cost.append((y, x))
                break

    goal_score = cost[0]
    return abs(goal_score[0] - 3) + abs(goal_score[1] - 1)
