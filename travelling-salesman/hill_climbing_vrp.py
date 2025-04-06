import numpy as np
import matplotlib.pyplot as plt
import time
import random
import os
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
    for i in range(len(path)):
        city1 = cities[path[i]]
        city2 = cities[path[(i+1) % len(path)]]
        total_distance += calculate_distance(city1, city2)
    return total_distance

def generate_greedy_solution(cities):
    num_cities = len(cities)
    start_city = random.randint(0, num_cities - 1)
    solution = [start_city]
    unvisited = set(range(num_cities))
    unvisited.remove(start_city)
    current_city = start_city

    while unvisited:
        nearest_city = min(unvisited, key=lambda city: calculate_distance(cities[current_city], cities[city]))
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
                return neighbor, neighbor_distance

    return best_neighbor, best_distance

def hill_climbing(cities, max_time_seconds):
    start_time = time.time()
    current_solution = generate_greedy_solution(cities)
    current_distance = calculate_path_distance(current_solution, cities)
    best_solution = current_solution.copy()
    best_distance = current_distance

    iterations = 0
    no_improve_count = 0
    optimal_time = 0

    while (time.time() - start_time) < max_time_seconds:
        iterations += 1
        neighbor, neighbor_distance = get_best_neighbor(current_solution, cities)

        if neighbor_distance < best_distance:
            best_distance = neighbor_distance
            best_solution = neighbor.copy()
            no_improve_count = 0
            optimal_time = time.time() - start_time
        else:
            no_improve_count += 1
            if no_improve_count >= 4:
                break

        current_solution = neighbor

    total_time = time.time() - start_time
    return best_solution, best_distance, total_time, optimal_time

def run_multiple_trials(cities, num_trials=5, time_per_trial=10):
    final_costs = []
    total_times = []
    optimal_times = []

    for trial in range(num_trials):
        print(f"\nTrial {trial+1}/{num_trials}")
        solution, distance, total_time, optimal_time = hill_climbing(cities, time_per_trial)
        print(f"Trial {trial+1}: Best Distance = {distance:.2f}, Total Time = {total_time:.2f}s, Optimal Time = {optimal_time:.2f}s")

        final_costs.append(distance)
        total_times.append(total_time)
        optimal_times.append(optimal_time)

    return final_costs, total_times, optimal_times

def plot_cost_and_time(costs, optimal_times):
    trials = np.arange(1, len(costs) + 1)
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(trials - width/2, costs, width, label='Cost', color='skyblue')
    bars2 = ax2.bar(trials + width/2, optimal_times, width, label='Optimal Time (s)', color='orange')

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')

    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Cost')
    ax2.set_ylabel('Optimal Time (s)')
    ax1.set_title('TSP Hill Climbing: Cost and Optimal Time per Trial')
    fig.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("hill_climbing_cost_vs_time.png")
    plt.close()

if __name__ == '__main__':
    filename = 'testcase.txt'
    if not os.path.exists(filename):
        print(f"Error: {filename} not found")
        exit(1)

    cities = read_city_data(filename)
    print(f"Loaded {len(cities)} cities")

    final_costs, total_times, optimal_times = run_multiple_trials(cities, num_trials=5, time_per_trial=30)

    plot_cost_and_time(final_costs, optimal_times)
    print("\nGraph saved as hill_climbing_cost_vs_time.png")
