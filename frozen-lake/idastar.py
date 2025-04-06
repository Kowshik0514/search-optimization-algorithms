import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import imageio
import time
from heapq import heappush, heappop
def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)
def state_to_coords(state, size):
    row = state // size
    col = state % size
    return row, col
def coords_to_state(row, col, size):
    return row * size + col
class IDAStarSolver:
    def __init__(self, env, size, render=False, timeout_sec=60):
        self.env = env
        self.size = size
        self.render_mode = render
        self.frames = []
        self.goal = (size - 1, size - 1)
        self.timeout_sec = timeout_sec
        self.lake_map = []
        for row in self.env.unwrapped.desc:
            self.lake_map.append([c.decode('utf-8') for c in row])
    def record_frame(self):
        if self.render_mode:
            frame = self.env.render()
            self.frames.append(frame)
    def is_valid(self, row, col):
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        cell = self.lake_map[row][col]
        return cell != 'H'
    def search(self, state, g, threshold, path, start_time):
        now = time.time()
        if now - start_time > self.timeout_sec:
            return float('inf'), False
        row, col = state_to_coords(state, self.size)
        h = manhattan_distance(row, col, self.goal[0], self.goal[1])
        f = g + h
        if f > threshold:
            return f, False
        if row == self.goal[0] and col == self.goal[1]:
            return f, True
        min_threshold = float('inf')
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        actions = [2, 1, 0, 3]
        for (dr, dc), action in zip(directions, actions):
            nr, nc = row + dr, col + dc
            if self.is_valid(nr, nc):
                new_state = coords_to_state(nr, nc, self.size)
                if new_state in path:
                    continue
                path.append(new_state)
                self.env.unwrapped.s = new_state
                self.record_frame()
                t, found = self.search(new_state, g + 1, threshold, path, start_time)
                if found:
                    return t, True
                if t < min_threshold:
                    min_threshold = t
                path.pop()
        return min_threshold, False
    def ida_star(self):
        start_time = time.time()
        self.frames = []
        path = []
        start = coords_to_state(0, 0, self.size)
        path.append(start)
        self.env.unwrapped.s = start
        self.record_frame()
        threshold = manhattan_distance(0, 0, self.goal[0], self.goal[1])
        while True:
            t, found = self.search(start, 0, threshold, path, start_time)
            if found:
                return (time.time() - start_time) * 1000, True, path
            if t == float('inf') or (time.time() - start_time) > self.timeout_sec:
                return (time.time() - start_time) * 1000, False, path
            threshold = t
def run_experiment(map_name="4x4", num_runs=5, timeout_min=1):
    times = []
    success_count = 0
    if map_name == "4x4":
        grid_size = 4
    elif map_name == "8x8":
        grid_size = 8
    else:
        raise ValueError("Unsupported map size. Use '4x4' or '8x8'.")
    for run in range(num_runs + 1):
        print(f"\n=== Run {run} ===")
        render_mode = "rgb_array" if run == 0 else None
        env = gym.make(
            "FrozenLake-v1",
            map_name=map_name,
            is_slippery=False,
            render_mode=render_mode
        )
        env.reset()
        if run == 0:
            print("\n=== Default Lake Map ===")
            for row in env.unwrapped.desc:
                print(" ".join([c.decode('utf-8') for c in row]))
            print("\n")
        solver = IDAStarSolver(
            env,
            grid_size,
            render=(run == 0),
            timeout_sec=timeout_min * 60
        )
        elapsed_time, success, path = solver.ida_star()
        status = "Reached Goal" if success else "Time Limit Exceeded or No Path"
        print(f"Time: {elapsed_time:.2f} ms | {status}")
        if success:
            success_count += 1
            print(f"Path length: {len(path)}")
            print(f"Path (states): {path}")
            path_coords = [state_to_coords(s, grid_size) for s in path]
            print(f"Path (coords): {path_coords}")
        if run == 0 and solver.frames:
            gif_name = "idastar_solution.gif" if success else "idastar_no_path.gif"
            imageio.mimsave(gif_name, solver.frames, fps=2)
            print(f"GIF saved: {gif_name}")
        if run > 0:
            times.append(elapsed_time)
    if times:
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, num_runs + 1), times, color='skyblue')
        plt.plot(range(1, num_runs + 1), times, marker='o', linestyle='-', color='blue')
        avg_time = sum(times)/len(times)
        plt.axhline(y=avg_time, color='r', linestyle='--',
                    label=f'Avg: {avg_time:.2f} ms')
        plt.title(f"IDA* Performance on FrozenLake {map_name} (τ = {timeout_min} min)")
        plt.xlabel("Run Number")
        plt.ylabel("Time (ms)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("idastar_times.png")
        success_rate = (success_count / num_runs) * 100 if num_runs > 0 else 0
        print(f"\n=== Summary ===")
        print(f"Average Time: {avg_time:.2f} ms")
        print(f"Success Rate: {success_rate:.1f}% ({success_count}/{num_runs})")
    return times, success_rate
if __name__ == "__main__":
    run_experiment(map_name="4x4", num_runs=5, timeout_min=1)
