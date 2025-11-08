import heapq
import numpy as np
from typing import List, Dict, Any
import time
import sys
from pathlib import Path

# Ensure src is in path for imports
_current_file = Path(__file__).resolve()
_src_dir = _current_file.parent.parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from problems.discrete.tsp import TSPProblem

"""
Implementation of A* search algorithm for solving TSP, using MST heuristic.
"""

class TSPState:
    def __init__(self, path: List[int], cost: float, remaining: set):
        self.path = path.copy()
        self.cost = cost
        self.remaining = remaining.copy()
        self.current_city = path[-1] if path else 0
        
    def __lt__(self, other):
        return self.cost < other.cost

class TSPAStarSolver:
    def __init__(self, problem: 'TSPProblem', time_limit: float = 3600.0):
        self.problem = problem
        self.distance_matrix = problem.distance_matrix
        self.n = problem.get_dim()
        self.time_limit = time_limit
        self.start_time = 0
        
    def solve(self, start_city: int = 0) -> Dict[str, Any]:
        self.start_time = time.time()
        
        initial_path = [start_city]
        initial_cost = 0.0
        remaining = set(range(self.n)) - {start_city}
        initial_state = TSPState(initial_path, initial_cost, remaining)
        
        open_set = []
        f_cost = initial_cost + self._heuristic(initial_state)
        state_id_counter = 0
        heapq.heappush(open_set, (f_cost, state_id_counter, initial_state))
        
        best_solution = None
        best_cost = float('inf')
        
        while open_set and time.time() - self.start_time < self.time_limit:
            f_cost, _, state = heapq.heappop(open_set)
            
            if f_cost >= best_cost:
                continue
            
            if not state.remaining:
                complete_cost = state.cost + self.distance_matrix[state.current_city, start_city]
                if complete_cost < best_cost:
                    best_cost = complete_cost
                    best_solution = state.path + [start_city]
                    open_set = [item for item in open_set if item[0] < best_cost]
                    heapq.heapify(open_set)
                continue
            
            for next_city in state.remaining:
                new_cost = state.cost + self.distance_matrix[state.current_city, next_city]
                
                if new_cost >= best_cost:
                    continue
                
                new_path = state.path + [next_city]
                new_remaining = state.remaining - {next_city}
                new_state = TSPState(new_path, new_cost, new_remaining)
                
                new_f_cost = new_cost + self._heuristic(new_state)
                
                if new_f_cost >= best_cost:
                    continue
                
                state_id_counter += 1
                heapq.heappush(open_set, (new_f_cost, state_id_counter, new_state))
        
        timed_out = time.time() - self.start_time >= self.time_limit
        
        return {
            'tour': best_solution,
            'cost': best_cost if best_solution else float('inf'),
            'time_elapsed': time.time() - self.start_time,
            'optimal': len(open_set) == 0 and not timed_out and best_cost < float('inf'),
            'timed_out': timed_out
        }
    
    def _heuristic(self, state: TSPState):
        return self._mst_heuristic(state)

    def _mst_heuristic(self, state: TSPState):
        if not state.remaining:
            return self.distance_matrix[state.current_city, state.path[0]]
            
        remaining_cities = list(state.remaining)
        current_city = state.current_city
        start_city = state.path[0]
        
        if not remaining_cities:
            return self.distance_matrix[current_city, start_city]
        
        mst_cost = self._prim_mst(remaining_cities)
        
        min_edge_from_current = min(self.distance_matrix[current_city, city] 
                                    for city in remaining_cities)
        
        min_edge_to_start = min(self.distance_matrix[city, start_city] 
                                for city in remaining_cities)
        
        return mst_cost + min_edge_from_current + min_edge_to_start
    
    def _prim_mst(self, cities: List[int]):
        if len(cities) <= 1:
            return 0
            
        key = {city: float('inf') for city in cities}
        parent = {city: None for city in cities}
        in_mst = {city: False for city in cities}
        
        key[cities[0]] = 0
        
        pq = [(0, cities[0])]
        total_cost = 0.0
        
        while pq:
            cost, u = heapq.heappop(pq)
            
            if in_mst[u]:
                continue
                
            in_mst[u] = True
            total_cost += cost
            
            for v in cities:
                if u == v:
                    continue
                
                weight = self.distance_matrix[u, v]
                if not in_mst[v] and weight < key[v]:
                    key[v] = weight
                    parent[v] = u
                    heapq.heappush(pq, (weight, v))
                    
        return total_cost