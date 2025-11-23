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
from agent import *

class TwoPhaseScheduler:
    """Deterministic order; runs all agents' step() then advance()."""
    def __init__(self, model):
        self.model = model
        self.agents = []

    def add(self, agent):
        self.agents.append(agent)

    def step(self):
        # phase 1: decisions
        for a in list(self.agents):
            a.step()
        # phase 2: updates
        for a in list(self.agents):
            adv = getattr(a, "advance", None)
            if callable(adv):
                adv()

class RandomTwoPhaseScheduler(TwoPhaseScheduler):
    """Same as above, but randomizes order each tick."""
    def step(self):
        order = list(self.agents)
        random.shuffle(order)
        for a in order:
            a.step()
        # keep same update order (or shuffle again if you prefer)
        for a in order:
            adv = getattr(a, "advance", None)
            if callable(adv):
                adv()
@dataclass
class ModelConfig:
    step_minutes: int = 1
    T_steps: int = 6 * 60          # 6 hours at 1-min steps
    rain_inflow: float = 0.01      # m per min added (tapers)
    initial_flood: float = 0.20
    pump_rate: float = 0.02
    ps_fail_threshold: float = 0.5
    ps_repair_time: int = 30
    bus_headway_min: int = 15
    base_bus_kmh: float = 25.0


class CityGraphModel(Model):
    def __init__(self, stops_gdf: gpd.GeoDataFrame, cfg: ModelConfig = ModelConfig(), seed=42):
        super().__init__(seed=seed)
        self.random = random.Random(seed)
        self.cfg = cfg
        self.schedule = TwoPhaseScheduler(self)

        # 1) Build graph from stops
        self.G = nx.Graph()
        # add nodes
        for i, row in stops_gdf.reset_index(drop=True).iterrows():
            pt = row.geometry
            self.G.add_node(i, x=float(pt.x), y=float(pt.y))
        # add edges: k-NN scaffolding for connectivity (k=2)
        k = 2
        coords = np.array([(self.G.nodes[n]["x"], self.G.nodes[n]["y"]) for n in self.G.nodes()])
        for i in range(len(coords)):
            d = np.linalg.norm(coords - coords[i], axis=1)
            nbrs = np.argsort(d)[1:k+1]
            for j in nbrs:
                if i != j:
                    if not self.G.has_edge(i, j):
                        self.G.add_edge(i, j)
        # edge lengths
        for u, v in self.G.edges():
            p1 = (self.G.nodes[u]["x"], self.G.nodes[u]["y"])
            p2 = (self.G.nodes[v]["x"], self.G.nodes[v]["y"])
            self.G.edges[u, v]["length_m"] = euclid(p1, p2) * 100000.0  # crude scale for meters
            self.G.edges[u, v]["travel_wt"] = self.G.edges[u, v]["length_m"]

        self.nodes = list(self.G.nodes())
        self.N = len(self.nodes)

        # 2) Flood states (per node)
        self.node_flood = {n: self.cfg.initial_flood for n in self.nodes}

        # 3) Choose special nodes (depot / power / pump) — you can set them explicitly
        self.depot_node = self.nodes[0]
        self.ps_node = self.nodes[len(self.nodes)//2]
        self.pump_node = self.nodes[min(len(self.nodes)-1, len(self.nodes)//2 - 1)]

        # 4) Agents: Power Station, Pump, Crew
        self.power_station = PowerStation("PS", self, self.ps_node,
                                          fail_threshold=self.cfg.ps_fail_threshold,
                                          repair_time=self.cfg.ps_repair_time)
        self.schedule.add(self.power_station)

        self.pump = Pump("PUMP", self, self.pump_node, pump_rate=self.cfg.pump_rate, radius_hops=2)
        self.schedule.add(self.pump)

        self.crew = RepairCrew("CREW", self, speed_edges_per_min=0.6)
        self.schedule.add(self.crew)

        # 5) Define 4 routes (lists of node ids). Replace this with your real routes if you have them.
        self.routes = self._make_four_routes()

        # 6) Spawn buses according to 15-min headway (pre-create with staggered starts)
        self.buses = []
        bid = 0
        for r_idx, route in enumerate(self.routes):
            # create departures staggered modulo headway
            for offset in range(0, 60, self.cfg.bus_headway_min):  # one hour worth of departures
                bus = Bus(f"BUS_{r_idx}_{offset}_{bid}", self, route,
                          headway_min=self.cfg.bus_headway_min, base_kmh=self.cfg.base_bus_kmh)
                # place bus at start with phase offset (progress along first edge)
                bus.progress = (offset / self.cfg.bus_headway_min) * (1.0 / max(1, len(route)))
                # randomize starting edge index slightly
                bus.idx = 0
                bus.next_idx = 1
                self.schedule.add(bus)
                self.buses.append(bus)
                bid += 1

        # 7) Data collector
        self.t = 0
        self.datacollector = DataCollector(
            model_reporters={
                "mean_flood": lambda m: float(np.mean(list(m.node_flood.values()))),
                "ps_up": lambda m: int(m.power_station.up),
                "service_active": lambda m: sum(1 for b in m.buses if b.active),
                "mean_bus_delay": lambda m: float(np.mean([b.delay for b in m.buses])) if m.buses else 0.0
            }
        )

    # --- Route builder (replace with your real routes if available) ---
    def _make_four_routes(self):
        """
        Create 4 simple loop routes from the node set:
        - sort nodes along x then split into 4 chunks, then each route walks the chunk with k-NN stitching.
        Replace this with real route definitions if you have them.
        """
        xs = [(n, self.G.nodes[n]["x"]) for n in self.nodes]
        xs.sort(key=lambda t: t[1])
        chunks = np.array_split([n for n, _ in xs], 4)

        routes = []
        for chunk in chunks:
            chunk = list(chunk)
            # order within chunk by nearest-neighbor greedy
            route = [chunk[0]]
            remaining = set(chunk[1:])
            while remaining:
                last = route[-1]
                nxt = min(remaining, key=lambda j: euclid(
                    (self.G.nodes[last]["x"], self.G.nodes[last]["y"]),
                    (self.G.nodes[j]["x"], self.G.nodes[j]["y"])))
                route.append(nxt)
                remaining.remove(nxt)
            # close the loop by ensuring consecutive nodes are connected (add edges if missing)
            for i in range(len(route)):
                u = route[i]
                v = route[(i+1) % len(route)]
                if not self.G.has_edge(u, v):
                    self.G.add_edge(u, v)
                    p1 = (self.G.nodes[u]["x"], self.G.nodes[u]["y"])
                    p2 = (self.G.nodes[v]["x"], self.G.nodes[v]["y"])
                    self.G.edges[u, v]["length_m"] = euclid(p1, p2) * 100000.0
                    self.G.edges[u, v]["travel_wt"] = self.G.edges[u, v]["length_m"]
            routes.append(route)
        return routes

    def step(self):
        # Hazard: time-varying rainfall (tapers)
        decay = max(0.0, 1.0 - self.t / max(1, int(0.7 * self.cfg.T_steps)))
        rain = self.cfg.rain_inflow * decay
        for n in self.nodes:
            self.node_flood[n] += rain

        # Update dynamic edge weights for pathfinding (optional; keep static for speed)
        # for u, v in self.G.edges():
        #     depth = 0.5*(self.node_flood[u] + self.node_flood[v])
        #     factor = 1.0 / max(0.25, flood_speed_factor(depth))
        #     self.G.edges[u, v]["travel_wt"] = self.G.edges[u, v]["length_m"] * factor

        self.schedule.step()
        self.datacollector.collect(self)
        self.t += 1

    def run(self):
        for _ in range(self.cfg.T_steps):
            self.step()