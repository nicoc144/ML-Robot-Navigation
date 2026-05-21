"""BFS path planner — guarantees fewest-steps (shortest) path."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

from .planner import Planner, Cell


class BFSPlanner(Planner):
    """Finds the shortest path (fewest cells) through a maze using BFS."""

    def find_path(self, start: Cell, goal: Cell) -> Optional[List[Cell]]:
        """Return cell path from *start* to *goal* (both inclusive), or ``None``."""
        came_from: Dict[Cell, Optional[Cell]] = {start: None}
        queue: deque[Cell] = deque([start])

        while queue:
            current = queue.popleft()

            if current == goal:
                return self._reconstruct(came_from, current)

            for neighbour in self._neighbours(current):
                if neighbour not in came_from:
                    came_from[neighbour] = current
                    queue.append(neighbour)

        return None
