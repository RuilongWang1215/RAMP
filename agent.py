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
# Helpers
# -------------------------
def flood_speed_factor(depth, d1=0.10, d2=0.40, min_factor=0.25):
    pass

def euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])
# -------------------------
# Agents
# -------------------------

class PowerStation(Agent):
    def __init__(self, unique_id, model, node, fail_threshold=0.5, repair_time=30):
        super().__init__(unique_id, model)
        self.node = node
        self.up = True
        self.fail_threshold = fail_threshold
        self.repair_time = repair_time
        self._repair_counter = 0

    def step(self):
        depth = self.model.node_flood[self.node]
        if self.up and depth >= self.fail_threshold:
            p_fail = min(1.0, 0.15 + 1.1*(depth - self.fail_threshold))
            if random.random() < p_fail:
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


class Pump(Agent):
    def __init__(self, unique_id, model, node, pump_rate=0.02, radius_hops=2):
        super().__init__(unique_id, model)
        self.node = node
        self.rate = pump_rate
        self.radius = radius_hops

    def step(self):
        pass

    def advance(self):
        # Pump runs only if power is up
        if not self.model.power_station.up:
            return
        # Reduce flood in a neighborhood around the pump node
        targets = nx.single_source_shortest_path_length(self.model.G, self.node, cutoff=self.radius).keys()
        for n in targets:
            self.model.node_flood[n] = max(0.0, self.model.node_flood[n] - self.rate)


class RepairCrew(Agent):
    def __init__(self, unique_id, model, speed_edges_per_min=0.5):
        super().__init__(unique_id, model)
        # On graph: at a node
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
        super().__init__(unique_id, model)
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