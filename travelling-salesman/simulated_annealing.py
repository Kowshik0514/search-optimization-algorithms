import math
import random
import time
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

def load_cities(filename):
    cities = []
    with open(filename, 'r') as f:
        for line in f:
            _, x, y = map(float, line.strip().split())
            cities.append((x, y))
    return cities

def calculate_path_distance(path, cities):
    distance = 0.0
    for i in range(len(path)):
        city1 = cities[path[i]]
        city2 = cities[path[(i + 1) % len(path)]]  # wrap around
        distance += math.dist(city1, city2)
    return distance

def random_solution(cities):
    solution = list(range(len(cities)))
    random.shuffle(solution)
    return solution

def three_opt_swap(route):
    a, b, c = sorted(random.sample(range(len(route)), 3))
    new_route = route[:a] + route[b:c] + route[a:b] + route[c:]
    return new_route

def draw_path(path, cities, filename):
    xs = [cities[i][0] for i in path + [path[0]]]
    ys = [cities[i][1] for i in path + [path[0]]]
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, 'o-', color='blue')
    plt.title('TSP Path')
    plt.axis('equal')
    plt.savefig(filename)
    plt.close()

def tsp_simulated_annealing(cities, temp=10000.0, cool=0.9995, print_interval=1000, run=1):
    current_sol = random_solution(cities)
    best_sol = list(current_sol)
    current_cost = calculate_path_distance(current_sol, cities)
    best_cost = current_cost
    print(f"Initial random cost: {current_cost:.2f}")

    iteration = 0
    frames = []

    while temp > 0.1:
        candidate = three_opt_swap(current_sol)
        candidate_cost = calculate_path_distance(candidate, cities)

        delta = candidate_cost - current_cost
        acceptance_prob = math.exp(-delta / temp) if delta > 0 else 1.0

        if candidate_cost < current_cost or random.random() < acceptance_prob:
            current_sol = candidate
            current_cost = candidate_cost

            if current_cost < best_cost:
                best_sol = list(current_sol)
                best_cost = current_cost

        iteration += 1
        if iteration % print_interval == 0:
            print(f"Iter {iteration} | Temp: {temp:.2f} | Current: {current_cost:.2f} | Best: {best_cost:.2f}")
            frame_file = f'frame_{run}_{iteration}.png'
            draw_path(best_sol, cities, frame_file)
            frames.append(frame_file)

        temp *= cool

    return best_sol, best_cost, frames

# === Main script ===
if __name__ == '__main__':
    cities = load_cities('./testcase.txt')
    print(f"Loaded {len(cities)} cities")

    all_runs_data = []
    best_overall_cost = float('inf')
    best_overall_path = None
    best_overall_frames = []
    best_run_index = -1

    total_time = 0.0

    for run in range(1, 6):
        print(f"\n=== Run {run} ===")
        start_time = time.time()

        best_path, best_cost, frames = tsp_simulated_annealing(cities, run=run)

        elapsed = time.time() - start_time
        total_time += elapsed

        print(f"Run {run} - Best cost: {best_cost:.2f}, Time: {elapsed:.4f} seconds")

        all_runs_data.append((run, best_cost, frames, elapsed))

        if best_cost < best_overall_cost:
            best_overall_cost = best_cost
            best_overall_path = best_path
            best_overall_frames = frames
            best_run_index = run

    avg_time = total_time / 5
    print(f"\nAverage execution time over 5 runs: {avg_time:.4f} seconds")
    print(f"Best overall cost found: {best_overall_cost:.2f} (from run {best_run_index})")

    # Save GIF from best run
    gif_filename = 'tsp_simulated_annealing.gif'
    images = [imageio.imread(frame) for frame in best_overall_frames if os.path.exists(frame)]
    imageio.mimsave(gif_filename, images, duration=0.4)
    print(f"GIF saved as {gif_filename}")

    # Clean up: delete non-best frames
    for run, _, frames, _ in all_runs_data:
        if run != best_run_index:
            for frame in frames:
                if os.path.exists(frame):
                    os.remove(frame)

    for frame in best_overall_frames:
        if os.path.exists(frame):
            os.remove(frame)

    # Plot time vs cost graph
    # Plot bar graph of time and cost
# Plot bar graph of time and cost
    run_numbers = [f'Run {run}' for run, _, _, _ in all_runs_data]
    costs = [cost for _, cost, _, _ in all_runs_data]
    times = [elapsed for _, _, _, elapsed in all_runs_data]

    x = range(len(run_numbers))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bar1 = plt.bar([i - width/2 for i in x], costs, width=width, label='Cost', color='skyblue')
    bar2 = plt.bar([i + width/2 for i in x], times, width=width, label='Time (s)', color='lightgreen')

    plt.xticks(ticks=x, labels=run_numbers)
    plt.xlabel('Run')
    plt.ylabel('Value')
    plt.title('TSP Simulated Annealing: Cost and Time per Run')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Add values on top of bars
    for rect in bar1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2, height + 1, f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

    for rect in bar2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2, height + 0.01, f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("tsp_simulated_annealing_stats_bar.png")
    plt.show()
