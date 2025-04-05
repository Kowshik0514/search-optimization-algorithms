def heuristic(state, n):
    return abs(state % n -(n -1)) + abs(state // n - (n - 1))