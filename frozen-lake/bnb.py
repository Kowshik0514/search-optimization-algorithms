import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import imageio
import time
import random
from heapq import heappush, heappop
from map_gen import create_map_with_path, create_map_with_no_path

def create_random_lake(size):
    lake = [[random.choice(['F'] * 8 + ['H'] * 2) for _ in range(size)] for _ in range(size)]
    lake[0][0] = 'S'
    lake[size-1][size-1] = 'G'
    return lake

def heiristic(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

class CustomDFBnB:
    def __init__(self, env, size, render=False):
        self.env = env
        self.size = size
        self.goal_row, self.goal_col = size-1, size-1
        self.best_cost = float('inf')
        self.best_path = []
        self.render_mode = render
        self.frames = []
        self.state_history = []
        self.goal_reached = False

    def is_valid_position(self, row, col, lake_map):
        if 0 <= row < self.size and 0 <= col < self.size:
            return lake_map[row][col] != 'H'
        return False

    def record_frame(self):
        if self.render_mode:
            frame = self.env.render()
            self.frames.append(frame)
            self.state_history.append(self.env.unwrapped.s)

    def solve(self, lake_map):
        self.best_cost = float('inf')
        self.best_path = []
        self.frames = []
        self.state_history = []
        self.goal_reached = False
        visited = set()  
        path_stack = []  
        goal_frame_index = -1  
        start_time = time.perf_counter_ns()
        if self.render_mode:
            self.record_frame()
        def dfbnb_recursive(state, current_cost):
            nonlocal goal_frame_index
            row, col = state // self.size, state % self.size
            self.env.unwrapped.s = state
            if self.render_mode:
                self.record_frame()
            if lake_map[row][col] == 'G':
                if current_cost < self.best_cost:
                    self.best_cost = current_cost
                    self.best_path = path_stack.copy()
                    self.goal_reached = True
                    if goal_frame_index == -1:  
                        goal_frame_index = len(self.frames) - 1
                return
            directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            action_indices = [2, 1, 0, 3]  
            neighbors = []
            for i, (dr, dc) in enumerate(directions):
                new_row, new_col = row + dr, col + dc
                if self.is_valid_position(new_row, new_col, lake_map):
                    new_state = new_row * self.size + new_col
                    if new_state not in visited:
                        h_cost = heiristic(new_row, new_col, self.goal_row, self.goal_col)
                        f_cost = (current_cost + 1) + h_cost
                        if f_cost < self.best_cost:
                            heappush(neighbors, (f_cost, action_indices[i], new_state))

            while neighbors:
                _, action, next_state = heappop(neighbors)
                path_stack.append(action)
                visited.add(next_state)
                dfbnb_recursive(next_state, current_cost + 1)
                path_stack.pop()
                visited.remove(next_state)

        initial_state = 0
        visited.add(initial_state)
        dfbnb_recursive(initial_state, 0)
        visited.remove(initial_state)
        end_time = time.perf_counter_ns()
        elapsed_ms = (end_time - start_time) / 1_000_000

        if self.render_mode and self.frames:
            if goal_frame_index != -1:
                solution_frames = self.frames[:goal_frame_index+1]
                filename = "dfbnb_solution_path.gif"
            else:
                solution_frames = self.frames
                filename = "dfbnb_no_path_found.gif"
            imageio.mimsave(filename, solution_frames, fps=2)
            print(f"\nGIF saved: {filename}")
        return elapsed_ms, self.best_cost != float('inf'), self.best_path

def run(grid_size=4, num_runs=5):
    times = []
    lake_map = create_random_lake(grid_size)
    print("\n=== Generated Map ===")
    for row in lake_map:
        print(" ".join(row))
    print("\n")
    lake_bytes = [[c.encode('utf-8') for c in row] for row in lake_map]
    for run in range(num_runs + 1):  
        if(run == 0):
            print("=== First Run ===")
        else:
            print(f"\n=== Run {run} ===")
        render_mode = "rgb_array" if run == 0 else None
        env = gym.make("FrozenLake-v1", desc=lake_bytes, is_slippery=False, render_mode=render_mode)
        env.reset()
        solver = CustomDFBnB(env, grid_size, render=(run == 0))
        elapsed_time, success, path = solver.solve(lake_map)
        status = "Reached Goal" if success else "No Path"
        print(f"Time: {elapsed_time:.2f} ms | {status}")
        if success:
            print(f"Path length: {len(path)}")
            print(f"Path: {path}")
        if run > 0:
            times.append(elapsed_time)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_runs + 1), times, marker='o', linestyle='-', color='green')
    plt.title("Custom DFBnB Algorithm Performance (Same Map)")
    plt.xlabel("Run Number")
    plt.ylabel("Time (ms)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dfbnb_times.png")
    avg_time = sum(times) / len(times) if times else 0
    print(f"\nAverage Time: {avg_time:.6f} ms")
    return times

if __name__ == "__main__":
    run(grid_size=4, num_runs=5)