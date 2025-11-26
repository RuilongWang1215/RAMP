# Model Architecture & Agent Logic

## 📂 Code Structure

### 1. `agent.py`
Contains all agent logic and the configuration class.
* **`PumpCrewManager`**: (New) Controls a fleet of pump trucks.
* **`PowerStation`**: Passive agent that can fail if flooded.
* **`RepairCrew`**: Fixes the power station.
* **`Bus`**: Circulates on fixed routes (slowed by flood).
* **`ModelConfig`**: A dataclass containing all simulation parameters (speeds, rain rates, counts).

### 2. `model.py`
Contains the environment and scheduling.
* **`CityGraphModel`**: Sets up the NetworkX graph.
  * *Note:* I increased connectivity to **k=4** (Nearest Neighbors) so crews have alternative paths to diverge.
* **`DataCollector`**: Tracks metrics like `total_water_removed`, `mean_flood`, and crew states (pumping, moving, unloading).

---

## 🧠 How the Pump Agents Work

I replaced the single `Pump` agent with a **Manager pattern** to handle coordination and physics more realistically.

### 1. The Manager Pattern
The `PumpCrewManager` is a single Mesa Agent that holds a list of dictionaries representing individual crews.
* **Why?** It allows us to centralize decision-making (preventing two crews from claiming the same node) while keeping the schedule simple.

### 2. The Logic Cycle
Each crew member follows this Finite State Machine:

`IDLE` $\to$ `MOVING_TO_WORK` $\to$ `PUMPING` $\to$ `MOVING_TO_DEPOT` $\to$ `UNLOADING`

### 3. Key Fixes & Features

#### A. The "Reservation" System (Divergence)
Previously, all crews would swarm the single wettest node.
* **Fix:** The manager checks `_get_taken_targets()`.
* **Logic:** If Crew A targets Node 5, Node 5 is marked as "taken." Crew B will see this and skip Node 5, targeting the *next* wettest node (e.g., Node 12). This forces the crews to spread out across the map.

#### B. The "Accumulator" (Physics)
Previously, crews got "stuck" because the flood slowed them down to < 1 edge per minute.
* **Fix:** I added an `edge_progress` variable.
* **Logic:** If a crew can only move 0.125 edges per tick (due to deep water), they accumulate this progress. After 8 ticks, `edge_progress >= 1.0`, and they physically move to the next node.

#### C. Dispatch Strategies
You can toggle how crews choose targets in `ModelConfig`:
* **`NEAREST`**: Go to the closest available flooded node (Greedy).
* **`CRITICAL`**: Go to the flooded node with the highest **Betweenness Centrality** (keeping major intersections dry).
