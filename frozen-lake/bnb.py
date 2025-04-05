from queue import PriorityQueue
import gymnasium as gym
import time
from heuristic import heuristic

def visualize_path_on_map(env, path):
    direction_map = {
        0: -1,
        1: env.unwrapped.ncol,
        2: 1,
        3: -env.unwrapped.ncol
    }

    desc = env.unwrapped.desc.astype(str)
    nrow, ncol = env.unwrapped.nrow, env.unwrapped.ncol

    def index_to_coords(idx):
        return divmod(idx, ncol)

    current = 0
    path_indices = [current]

    for action in path:
        current += direction_map[action]
        path_indices.append(current)

    map_display = []
    for r in range(nrow):
        row = []
        for c in range(ncol):
            idx = r * ncol + c
            if desc[r, c] == 'S':
                row.append('S')
            elif desc[r, c] == 'G':
                row.append('G')
            elif desc[r, c] == 'H':
                row.append('H')
            elif idx in path_indices[1:-1]:
                row.append('*')
            else:
                row.append('.')
        map_display.append(row)

    print("\nMap with path:")
    for row in map_display:
        print(" ".join(row))

def branch_and_bound(env, n, max_time=600):
    start = time.time()
    visited = set()
    queue = PriorityQueue()

    state = env.reset()[0]
    queue.put((heuristic(state, n), 0, state, []))

    while not queue.empty() and (time.time() - start) < max_time:
        _, cost, current_state, path = queue.get()
        if current_state in visited:
            continue
        visited.add(current_state)
        if env.unwrapped.desc.reshape(-1)[current_state] == b'G':
            return path, len(path), time.time() - start
        for action in range(env.action_space.n):
            for prob, next_state, reward, done in env.unwrapped.P[current_state][action]:
                if prob > 0 and next_state not in visited:
                    h = heuristic(next_state, n)
                    queue.put((cost + 1 + h, cost + 1, next_state, path + [action]))

    return None, -1, time.time() - start

if __name__ == "__main__":
    n = 4
    map_name = f"{n}x{n}"
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=False)
    total_runs = 10
    for i in range(total_runs):
        print(f"Run {i + 1}/{total_runs}")
        path, steps, elapsed_time = branch_and_bound(env, n)
        if path is not None:
            print(f"Path found: {path}")
            print(f"Steps taken: {steps}")
            # visualize_path_on_map(env, path)
        else:
            print("No path found")
        print(f"Elapsed time: {elapsed_time:.8f} seconds\n")
