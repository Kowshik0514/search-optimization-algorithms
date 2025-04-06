import numpy as np
import matplotlib.pyplot as plt
import time
import random
import os
from matplotlib.animation import FuncAnimation
import imageio
import matplotlib

matplotlib.use('Agg')

def read_city_data(filename):
    cities = []
    with open(filename, 'r') as file:
        for line in file:

            if line.strip() == '' or line.strip().startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                city_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                cities.append((city_id, x, y))
    return cities

def calculate_distance(city1, city2):
    return np.sqrt((city1[1] - city2[1])**2 + (city1[2] - city2[2])**2)

def calculate_path_distance(path, cities):
    total_distance = 0
    for i in range(len(path) - 1):
        city1 = cities[path[i]]
        city2 = cities[path[i+1]]
        total_distance += calculate_distance(city1, city2)

    total_distance += calculate_distance(cities[path[-1]], cities[path[0]])
    return total_distance

def generate_greedy_solution(cities):
    num_cities = len(cities)

    start_city = random.randint(0, num_cities - 1)
    solution = [start_city]
    unvisited = set(range(num_cities))
    unvisited.remove(start_city)

    current_city = start_city

    while unvisited:
        min_distance = float('inf')
        nearest_city = None

        for city in unvisited:
            dist = calculate_distance(cities[current_city], cities[city])
            if dist < min_distance:
                min_distance = dist
                nearest_city = city

        solution.append(nearest_city)
        unvisited.remove(nearest_city)
        current_city = nearest_city

    return solution

def get_best_neighbor(solution, cities):
    best_neighbor = solution.copy()
    best_distance = calculate_path_distance(solution, cities)

    for i in range(len(solution)):
        for j in range(i+1, len(solution)):
            neighbor = solution.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

            neighbor_distance = calculate_path_distance(neighbor, cities)
            if neighbor_distance < best_distance:
                best_distance = neighbor_distance
                best_neighbor = neighbor.copy()
                return best_neighbor, best_distance  

    return best_neighbor, best_distance

def create_solution_frame(cities, path, distance, iteration, time_elapsed, frame_dir="frames"):

    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    fig, ax = plt.subplots(figsize=(10, 6))

    x_coords = [cities[i][1] for i in range(len(cities))]
    y_coords = [cities[i][2] for i in range(len(cities))]

    ax.scatter(x_coords, y_coords, c='red', s=50)

    for i in range(len(path)):
        city1 = cities[path[i]]
        city2 = cities[path[(i+1) % len(path)]]
        ax.plot([city1[1], city2[1]], [city1[2], city2[2]], 'b-')

    for i, city in enumerate(cities):
        ax.text(city[1], city[2], str(city[0]), fontsize=12)

    ax.set_title(f"TSP Solution - Iteration {iteration} (Distance: {distance:.2f}, Time: {time_elapsed:.2f}s)")
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True)

    frame_path = os.path.join(frame_dir, f"frame_{iteration:05d}.png")
    plt.savefig(frame_path)
    plt.close(fig)

    return frame_path

def hill_climbing(cities, max_time_seconds, capture_frames=True, frame_interval=10):
    start_time = time.time()

    current_solution = generate_greedy_solution(cities)
    current_distance = calculate_path_distance(current_solution, cities)

    best_solution = current_solution.copy()
    best_distance = current_distance

    iterations = 0
    frame_paths = []

    if capture_frames:
        frame_path = create_solution_frame(cities, current_solution, current_distance, iterations, 0)
        frame_paths.append(frame_path)

    while (time.time() - start_time) < max_time_seconds:
        iterations += 1

        neighbor, neighbor_distance = get_best_neighbor(current_solution, cities)

        if neighbor_distance >= current_distance:

            new_solution = generate_greedy_solution(cities)
            new_distance = calculate_path_distance(new_solution, cities)

            current_solution = new_solution
            current_distance = new_distance
        else:
            current_solution = neighbor
            current_distance = neighbor_distance

            if current_distance < best_distance:
                best_distance = current_distance
                best_solution = current_solution.copy()

                if capture_frames:
                    elapsed_time = time.time() - start_time
                    frame_path = create_solution_frame(cities, best_solution, best_distance, iterations, elapsed_time)
                    frame_paths.append(frame_path)

        if capture_frames and iterations % frame_interval == 0:
            elapsed_time = time.time() - start_time

            if current_solution == best_solution or iterations % (frame_interval * 5) == 0:
                frame_path = create_solution_frame(cities, current_solution, current_distance, iterations, elapsed_time)
                frame_paths.append(frame_path)

        if iterations % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"Iteration {iterations}: Current distance = {current_distance:.2f}, Best distance = {best_distance:.2f}, Time elapsed: {elapsed_time:.2f}s")

    total_time = time.time() - start_time
    print(f"Completed in {total_time:.2f} seconds")
    print(f"Completed {iterations} iterations")
    print(f"Best distance found: {best_distance:.2f}")

    if capture_frames:
        frame_path = create_solution_frame(cities, best_solution, best_distance, iterations, total_time)
        frame_paths.append(frame_path)

    return best_solution, best_distance, frame_paths

def run_multiple_trials(cities, num_trials=5, total_time_seconds=60):
    best_solutions = []
    best_distances = []
    all_frame_paths = []  

    time_per_trial = total_time_seconds / num_trials

    total_start_time = time.time()

    for trial in range(num_trials):
        print(f"\nTrial {trial+1}/{num_trials}")

        solution, distance, frame_paths = hill_climbing(cities, time_per_trial, capture_frames=True)

        best_solutions.append(solution)
        best_distances.append(distance)

        all_frame_paths.extend(frame_paths)

        print(f"Trial {trial+1} completed")
        print(f"Best distance found: {distance:.2f}")

    total_time = time.time() - total_start_time
    print(f"\nTotal time for all trials: {total_time:.2f} seconds")

    best_idx = np.argmin(best_distances)
    best_overall_solution = best_solutions[best_idx]
    best_overall_distance = best_distances[best_idx]

    avg_distance = np.mean(best_distances)
    std_distance = np.std(best_distances)

    print("\nSummary Statistics:")
    print(f"Average Distance: {avg_distance:.2f} ± {std_distance:.2f}")
    print(f"Best Solution Distance: {best_overall_distance:.2f}")

    create_gif(all_frame_paths, "tsp_combined_animation.gif", fps=3)

    return best_overall_solution, best_overall_distance, all_frame_paths

def create_gif(frame_paths, filename="tsp_animation.gif", fps=5):
    print(f"Creating GIF with {len(frame_paths)} frames...")

    images = []
    for frame_path in frame_paths:
        images.append(imageio.imread(frame_path))

    imageio.mimsave(filename, images, fps=fps)
    print(f"GIF saved as {filename}")

if __name__ == "__main__":

    filename = "testcase.txt"
    if not os.path.exists(filename):
        print(f"Error: {filename} not found")
        exit(1)

    cities = read_city_data(filename)
    print(f"Loaded {len(cities)} cities from {filename}")

    num_trials = 5
    total_time_seconds = 60  

    best_solution, best_distance, frame_paths = run_multiple_trials(
        cities, num_trials, total_time_seconds
    )

    plt.figure(figsize=(10, 6))

    x_coords = [cities[i][1] for i in range(len(cities))]
    y_coords = [cities[i][2] for i in range(len(cities))]

    plt.scatter(x_coords, y_coords, c='red', s=50)

    for i in range(len(best_solution)):
        city1 = cities[best_solution[i]]
        city2 = cities[best_solution[(i+1) % len(best_solution)]]
        plt.plot([city1[1], city2[1]], [city1[2], city2[2]], 'b-')

    for i, city in enumerate(cities):
        plt.text(city[1], city[2], str(city[0]), fontsize=12)

    plt.title(f"Optimal TSP Solution (Distance: {best_distance:.2f})")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.savefig("tsp_optimal_solution.png")
    plt.close()

    if frame_paths:
        create_gif(frame_paths, "tsp_animation.gif", fps=3)

    print("\nOutput files:")
    print("- tsp_optimal_solution.png")
    print("- tsp_animation.gif")