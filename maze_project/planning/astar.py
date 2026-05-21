"""A* path planner — inherits shared infrastructure from Planner."""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

from .planner import Planner, Cell


class AStarPlanner(Planner):
    """Finds the shortest path through a maze using A* with Manhattan distance."""

    def find_path(self, start: Cell, goal: Cell) -> Optional[List[Cell]]:
        """Return cell path from *start* to *goal* (both inclusive), or ``None``."""
        open_heap: List[Tuple[float, Cell]] = []
        heapq.heappush(open_heap, (0.0, start))

        came_from: Dict[Cell, Optional[Cell]] = {start: None}
        g_score:   Dict[Cell, float]          = {start: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current == goal:
                return self._reconstruct(came_from, current)

            for neighbour in self._neighbours(current):
                tentative_g = g_score[current] + 1.0
                if tentative_g < g_score.get(neighbour, float("inf")):
                    came_from[neighbour] = current
                    g_score[neighbour]   = tentative_g
                    f = tentative_g + self._heuristic(neighbour, goal)
                    heapq.heappush(open_heap, (f, neighbour))

        return None

    @staticmethod
    def _heuristic(a: Cell, b: Cell) -> float:
        """Manhattan distance heuristic."""
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))
