from dataclasses import dataclass
import random
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
import networkx as nx
from mesa import Model, Agent
from mesa.datacollection import DataCollector

# -------------------------
# Configuration
# -------------------------
@dataclass
class ModelConfig:
    step_minutes: int = 1
    T_steps: int = 6 * 60          # 6 hours
    rain_inflow: float = 0.001      
    initial_flood: float = 0.20
    
    # Power Station Config
    ps_fail_threshold: float = 0.5
    ps_repair_time: int = 30
    
    # Bus Config
    bus_headway_min: int = 15
    base_bus_kmh: float = 25.0
    
    # Pump Crew Config (New)
    num_pump_crews: int = 3
    tank_capacity_liters: float = 10000.0
    pump_rate_lpm: float = 2000.0
    crew_speed: float = 0.5        # Edges per minute
    node_area_m2: float = 400.0    # Area to convert depth to volume
    dispatch_strategy: str = 'NEAREST'

# -------------------------
# Helpers
# -------------------------
def flood_speed_factor(depth, d1=0.10, d2=0.40, min_factor=0.25):
    if depth <= d1:
        return 1.0  # No slowdown below 10cm
    elif depth >= d2:
        return min_factor  # Maximum slowdown above 40cm (25% speed)
    else:
        # Linear slowdown between 10cm and 40cm
        return 1.0 - (1.0 - min_factor) * ((depth - d1) / (d2 - d1))

def euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# -------------------------
# Agents
# -------------------------

class PowerStation(Agent):
    def __init__(self, unique_id, model, node, fail_threshold=0.5, repair_time=30):
        self.unique_id = unique_id
        self.model = model
        self.node = node
        self.up = True
        self.fail_threshold = fail_threshold
        self.repair_time = repair_time
        self._repair_counter = 0

    def step(self):
        depth = self.model.node_flood[self.node]
        if self.up and depth >= self.fail_threshold:
            p_fail = min(1.0, 0.15 + 1.1*(depth - self.fail_threshold))
            if self.model.random.random() < p_fail:
                print(f"!!! POWER STATION FAILED at Step {self.model.t} (Depth: {depth:.2f}m) !!!")
                self.up = False
                self._repair_counter = 0

    def advance(self):
        pass

    def begin_repair(self):
        self._repair_counter = self.repair_time

    def repair_tick(self):
        if self._repair_counter > 0:
            self._repair_counter -= 1
            if self._repair_counter == 0:
                self.up = True

class PumpCrewManager(Agent):
    def __init__(self, unique_id, model, cfg: ModelConfig):
        self.unique_id = unique_id
        self.model = model
        self.cfg = cfg
        
        # Pre-calculate Centrality
        self.centrality = nx.betweenness_centrality(model.G)
        
        self.crews = []
        for i in range(cfg.num_pump_crews):
            self.crews.append({
                'id': i,
                'node': model.depot_node,
                'load_liters': 0.0,
                'target': None,
                'path': [],
                'state': 'IDLE',
                'edge_progress': 0.0,
                
                # Staging
                'next_node': model.depot_node,
                'next_load': 0.0,
                'next_state': 'IDLE',
                'pump_amount': 0.0,
                'next_path': [],
                'next_target': None,
                'next_edge_progress': 0.0
            })

    def step(self):
        depot = self.model.depot_node

        for crew in self.crews:
            # 1. Sync Staging (Copy current to next)
            crew['next_node'] = crew['node']
            crew['next_load'] = crew['load_liters']
            crew['next_state'] = crew['state']
            crew['next_path'] = list(crew['path'])
            crew['next_target'] = crew['target'] # Default to current target
            crew['next_edge_progress'] = crew['edge_progress']
            crew['pump_amount'] = 0.0

            # 2. DECISION LOGIC
            
            # IDLE -> Find Work
            if crew['state'] == 'IDLE':
                if crew['load_liters'] > 0:
                    crew['next_state'] = 'UNLOADING'
                else:
                    target, path = self._find_new_target(crew)
                    if target is not None:
                        crew['next_target'] = target # Set INTENTION immediately
                        crew['next_path'] = path
                        crew['next_state'] = 'MOVING_TO_WORK'
                        crew['next_edge_progress'] = 0.0
                        print(f"--> Crew {crew['id']} selected NEW Target: {target}")

            # PUMPING -> Full?
            if crew['state'] == 'PUMPING' and crew['load_liters'] >= self.cfg.tank_capacity_liters:
                self._route_to_depot(crew, depot)

            # PUMPING -> Dry?
            if crew['state'] == 'PUMPING':
                depth = self.model.node_flood[crew['node']]
                if depth <= 0.01:
                    target, path = self._find_new_target(crew)
                    if target:
                        crew['next_target'] = target # Set INTENTION
                        crew['next_path'] = path
                        crew['next_state'] = 'MOVING_TO_WORK'
                        crew['next_edge_progress'] = 0.0
                        print(f"--> Crew {crew['id']} rerouting to: {target}")
                    else:
                        self._route_to_depot(crew, depot)

            # 3. MOVEMENT
            if crew['next_state'] in ['MOVING_TO_WORK', 'MOVING_TO_DEPOT']:
                self._calculate_move(crew)
                if len(crew['next_path']) <= 1 and crew['next_node'] == crew['next_target']:
                    if crew['next_state'] == 'MOVING_TO_WORK':
                        crew['next_state'] = 'PUMPING'
                    else:
                        crew['next_state'] = 'UNLOADING'

            # 4. ACTION
            if crew['next_state'] == 'PUMPING':
                self._calculate_pump(crew)
            elif crew['next_state'] == 'UNLOADING':
                unload_rate = self.cfg.pump_rate_lpm * 2
                crew['next_load'] = max(0, crew['load_liters'] - unload_rate)
                if crew['next_load'] == 0:
                    crew['next_state'] = 'IDLE'

    def advance(self):
        for crew in self.crews:
            crew['node'] = crew['next_node']
            crew['load_liters'] = crew['next_load']
            crew['state'] = crew['next_state']
            crew['path'] = crew['next_path']
            crew['target'] = crew['next_target']
            crew['edge_progress'] = crew['next_edge_progress']

            if crew['pump_amount'] > 0:
                node = crew['node']
                current_depth = self.model.node_flood[node]
                depth_reduction = crew['pump_amount'] / 1000.0 / self.cfg.node_area_m2
                self.model.node_flood[node] = max(0, current_depth - depth_reduction)

    # --- HELPERS ---

    def _get_taken_targets(self):
        """
        Returns a set of nodes currently targeted by ANY crew.
        CRITICAL FIX: Checks 'next_target' (Intention) to prevent race conditions.
        """
        taken = set()
        for c in self.crews:
            # If a crew just decided on a target in this step, 'next_target' holds it
            if c['next_target'] is not None:
                taken.add(c['next_target'])
            # Fallback to current target if no change
            elif c['target'] is not None:
                taken.add(c['target'])
        return taken

    def _find_new_target(self, crew):
        taken = self._get_taken_targets()
        
        candidates = [
            n for n in self.model.nodes 
            if self.model.node_flood[n] > 0.05 and n not in taken
        ]
        
        if not candidates:
            return None, []

        best_target = None
        best_path = []
        best_score = -float('inf') 

        for target in candidates:
            try:
                path = nx.shortest_path(self.model.G, crew['node'], target, weight="travel_wt")
                dist = len(path)
                flood_val = self.model.node_flood[target]
                
                if self.cfg.dispatch_strategy == 'CRITICAL':
                    # Score = (Flood * Centrality) / Distance
                    centrality_val = self.centrality[target] + 0.01
                    score = (flood_val * centrality_val) / max(1, dist)
                else: # NEAREST
                    # Score = Flood / Distance
                    score = flood_val / max(1, dist)

                if score > best_score:
                    best_score = score
                    best_target = target
                    best_path = path
                    
            except nx.NetworkXNoPath:
                continue
        
        return best_target, best_path

    def _calculate_move(self, crew):
        if not crew['next_path']: return
        lookahead = crew['next_path'][:3]
        depths = [self.model.node_flood[n] for n in lookahead]
        mean_depth = float(np.mean(depths)) if depths else 0.0
        factor = flood_speed_factor(mean_depth)
        
        dist = max(0.1, self.cfg.crew_speed * factor)
        crew['next_edge_progress'] += dist
        
        while crew['next_edge_progress'] >= 1.0 and len(crew['next_path']) > 1:
            crew['next_node'] = crew['next_path'].pop(1)
            crew['next_edge_progress'] -= 1.0

    def _calculate_pump(self, crew):
        node = crew['node']
        current_depth = self.model.node_flood[node]
        if current_depth <= 0: return
        water_on_node_liters = current_depth * self.cfg.node_area_m2 * 1000.0
        space_in_tank = self.cfg.tank_capacity_liters - crew['load_liters']
        liters_to_remove = min(water_on_node_liters, space_in_tank, self.cfg.pump_rate_lpm)
        crew['pump_amount'] = liters_to_remove
        crew['next_load'] += liters_to_remove

    def _route_to_depot(self, crew, depot):
        try:
            crew['next_path'] = nx.shortest_path(self.model.G, crew['node'], depot, weight="travel_wt")
            crew['next_target'] = depot
            crew['next_state'] = 'MOVING_TO_DEPOT'
            crew['next_edge_progress'] = 0.0
        except nx.NetworkXNoPath:
            crew['next_state'] = 'IDLE'


class RepairCrew(Agent):
    def __init__(self, unique_id, model, speed_edges_per_min=0.5):
        self.unique_id = unique_id
        self.model = model
        self.node = model.depot_node
        self.speed = speed_edges_per_min  # edges per minute (slowed by flood)
        self.path = []
        self.busy = False

    def step(self):
        ps = self.model.power_station
        if not ps.up and not self.busy:
            # compute shortest path to power station
            self.path = nx.shortest_path(self.model.G, self.node, ps.node, weight="travel_wt")
            self.busy = True

        if self.busy and self.path:
            # move along path based on slowed speed (consume edges)
            mean_depth = float(np.mean([self.model.node_flood[n] for n in self.path]))
            factor = flood_speed_factor(mean_depth)
            edges_to_advance = max(0.1, self.speed * factor)
            # advance across discrete nodes
            progress = edges_to_advance
            while progress > 0 and len(self.path) > 1:
                # Move one edge
                self.node = self.path.pop(1)
                progress -= 1.0

            # Arrived at PS node
            if len(self.path) <= 1 and self.node == ps.node:
                ps.begin_repair()
                # stay and tick repair until done
                ps.repair_tick()
                if ps.up:
                    # go back to depot
                    self.path = nx.shortest_path(self.model.G, self.node, self.model.depot_node, weight="travel_wt")
                    # set busy to False only after we arrive at depot
                    if len(self.path) <= 1:
                        self.busy = False
            # If still enroute and PS still down, keep moving next step
        else:
            # Idle: drift toward depot if not there
            if self.node != self.model.depot_node:
                self.path = nx.shortest_path(self.model.G, self.node, self.model.depot_node, weight="travel_wt")
                if len(self.path) > 1:
                    self.node = self.path.pop(1)

    def advance(self):
        pass


class Bus(Agent):
    def __init__(self, unique_id, model, route_nodes, headway_min=15, base_kmh=25.0):
        self.unique_id = unique_id
        self.model = model
        self.route = route_nodes  # list of node ids (loop)
        self.idx = 0              # current index on route
        self.next_idx = 1
        self.progress = 0.0       # progress along current edge [0,1]
        self.active = True
        self.headway_min = headway_min
        self.base_kmh = base_kmh
        self.delay = 0.0

    def current_edge(self):
        return (self.route[self.idx], self.route[self.next_idx])

    def step(self):
        if not self.active:
            return
        u, v = self.current_edge()
        # Edge length (meters) & travel speed (m/min)
        length_m = self.model.G.edges[u, v]["length_m"]
        # base speed -> m/min
        base_m_per_min = (self.base_kmh * 1000.0) / 60.0

        # Slowdown by flood at mid-edge
        depth_u = self.model.node_flood[u]
        depth_v = self.model.node_flood[v]
        depth_edge = 0.5 * (depth_u + depth_v)
        factor = flood_speed_factor(depth_edge)
        speed_m_per_min = max(0.5, base_m_per_min * factor)

        # increment progress
        dt = 1.0  # minute per step
        delta = (speed_m_per_min * dt) / max(1e-6, length_m)
        self.progress += delta
        # delay proxy: lost fraction vs free-flow (if factor<1)
        self.delay += (1.0 - factor)

        # advance to next edge if we reach the end
        while self.progress >= 1.0:
            self.idx = self.next_idx
            self.next_idx = (self.next_idx + 1) % len(self.route)
            self.progress -= 1.0

    def advance(self):
        pass