"""
Abstract base class for maze path planners.

All planners operate on logical (cell_row, cell_col) space, not the
full (2n+1) bitmap space.  Subclasses only need to implement find_path();
wall-aware neighbour enumeration and path reconstruction are provided here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

from maze_generator.maze_data import MazeData

# A cell in logical space
Cell = Tuple[int, int]


class Planner(ABC):
    """Interface for path planners that operate on a MazeData grid."""

    _DIRECTIONS: List[Tuple[int, int]] = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def __init__(self, maze_data: MazeData) -> None:
        self._maze = maze_data
        # Cache grid as numpy array for fast O(1) wall lookups
        self._grid = np.asarray(maze_data.grid, dtype=np.uint8)

    @abstractmethod
    def find_path(self, start: Cell, goal: Cell) -> Optional[List[Cell]]:
        """Return the list of logical cells from *start* to *goal*, or ``None``
        if no path exists.  The returned list includes both endpoints."""
        ...

    # ------------------------------------------------------------------
    # Shared helpers (available to all subclasses)
    # ------------------------------------------------------------------

    def _neighbours(self, cell: Cell) -> List[Cell]:
        """Return the logical neighbours of *cell* reachable without a wall."""
        row, col = cell
        md = self._maze
        result: List[Cell] = []
        for dr, dc in self._DIRECTIONS:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < md.height and 0 <= nc < md.width):
                continue
            if self._grid[2 * row + 1 + dr, 2 * col + 1 + dc] == 0:
                result.append((nr, nc))
        return result

    @staticmethod
    def _reconstruct(
        came_from: Dict[Cell, Optional[Cell]], current: Cell
    ) -> List[Cell]:
        """Trace *came_from* back from *current* to the start."""
        path: List[Cell] = []
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path
