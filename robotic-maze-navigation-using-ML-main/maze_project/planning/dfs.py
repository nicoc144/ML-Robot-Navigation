"""DFS path planner — finds *a* path, not necessarily the shortest one."""

from __future__ import annotations

from typing import Dict, List, Optional

from .planner import Planner, Cell


class DFSPlanner(Planner):
    """Finds a path through a maze using iterative DFS (non-recursive).

    Note: DFS does not guarantee the shortest path.
    """

    def find_path(self, start: Cell, goal: Cell) -> Optional[List[Cell]]:
        """Return cell path from *start* to *goal* (both inclusive), or ``None``."""
        came_from: Dict[Cell, Optional[Cell]] = {start: None}
        stack: List[Cell] = [start]

        while stack:
            current = stack.pop()

            if current == goal:
                return self._reconstruct(came_from, current)

            for neighbour in self._neighbours(current):
                if neighbour not in came_from:
                    came_from[neighbour] = current
                    stack.append(neighbour)

        return None
