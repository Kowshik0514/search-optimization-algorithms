# AI Assignment 2: Search and Optimization

This repository contains implementations of four search and optimization algorithms applied on the Frozen Lake and Traveling Salesman Problem (TSP) environments.


---

## 🧠 Algorithms Implemented

- **Branch and Bound (BnB)** – on Frozen Lake
- **Iterative Deepening A\* (IDA\*)** – on Frozen Lake
- **Hill Climbing (HC)** – on TSP
- **Simulated Annealing (SA)** – on TSP

---

## 🗂 Folder Structure

```
search-optimization-algorithms/
|
├── frozen-lake/
│   ├── idastar.py                # IDA* implementation
│   ├── bnb.py                    # Heuristic BnB implementation
│   ├── map_gen.py                # Map generator
│   └── *.png, *.gif              # Visualizations and performance plots
|
├── travelling-salesman/
│   ├── hill_climbing_vrp.py      # Hill Climbing for TSP
│   ├── simulated_annealing.py    # Simulated Annealing for TSP
│   └── *.png, *.gif              # Visualizations and stats
|
├── requirements.txt              # All dependencies
├── README.md                     # You are here
└── AI_Assignment_2.pptx          # Slide deck with results
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/search-optimization-algorithms.git
cd search-optimization-algorithms
```

### 2. Create & activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 🔸 Frozen Lake (BnB & IDA\*)

#### ➔ Heuristic Branch and Bound
```bash
cd frozen-lake
python bnb.py
```

#### ➔ Iterative Deepening A*
```bash
cd frozen-lake
python idastar.py
```

> Output: GIFs of solution paths + time vs performance plots.

---

### 🔶 Traveling Salesman Problem (Hill Climbing & Simulated Annealing)

#### ➔ Hill Climbing
```bash
cd travelling-salesman
python hill_climbing_vrp.py
```

#### ➔ Simulated Annealing
```bash
cd travelling-salesman
python simulated_annealing.py
```

> Output: Animation of TSP solution over time + cost vs time plots.

---

## 📊 Outputs

- ✅ GIF animations of algorithms progressing to solutions
- ✅ Time vs cost plots for convergence analysis
- ✅ `testcase.txt` and intermediate solution frames stored for debugging/analysis

---

## 📌 Notes

- All algorithms are tested on **5 runs** and capped at **10 minutes** runtime.
- Slide deck (`AI_Assignment_2.pptx`).

---