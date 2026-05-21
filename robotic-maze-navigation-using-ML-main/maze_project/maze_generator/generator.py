"""
MazeGenerator – creates perfect mazes using iterative Depth-First Search
(Randomised DFS / recursive-backtracker algorithm).

A *perfect* maze has exactly one path between any two cells, which
guarantees at least one solution between the entrance and the exit by
construction.

Entrance: north wall of cell (0, 0)              – top-left
Exit:     south wall of cell (height-1, width-1) – bottom-right
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np

from .maze_data import MazeData


class MazeGenerator:
    """Generate mazes and return them as :class:`MazeData` instances."""

    # Four cardinal directions as (Δrow, Δcol)
    _DIRECTIONS: List[Tuple[int, int]] = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Public API

    def generate(
        self,
        width: int,
        height: int,
        cell_size: float = 1.0,
        wall_width: float = 0.15,
        wall_height: float = 1.0,
        seed: Optional[int] = None,
    ) -> MazeData:
        """Generate a perfect maze and return it as a :class:`MazeData`.

        Parameters
        ----------
        width, height:
            Number of cells in each dimension (must be ≥ 2).
        cell_size:
            Physical size of one open cell in the simulation (metres).
        wall_width:
            Physical thickness of wall segments (metres).
        wall_height:
            Height of wall bodies in the simulation (metres).
        seed:
            Optional RNG seed for reproducibility.
        """
        if width < 2 or height < 2:
            raise ValueError("Maze must be at least 2×2 cells.")

        rng = random.Random(seed)

        # 1. Initialise a fully-walled grid with numpy
        grid_rows = 2 * height + 1
        grid_cols = 2 * width + 1
        grid = np.ones((grid_rows, grid_cols), dtype=np.uint8)

        # Open all room cells (odd row, odd col) in one vectorised step
        grid[1::2, 1::2] = 0

        # 2. Iterative DFS to carve passages
        visited = np.zeros((height, width), dtype=bool)
        self._carve_passages(0, 0, width, height, visited, grid, rng)

        # 3. Open entrance and exit in the boundary walls
        # Entrance: north wall of cell (0, 0)  →  grid[0][1]
        grid[0][1] = 0
        entrance = {"cell_row": 0, "cell_col": 0, "side": "north"}

        # Exit: south wall of cell (height-1, width-1)  →  grid[2*height][2*(width-1)+1]
        grid[2 * height][2 * (width - 1) + 1] = 0
        exit_ = {"cell_row": height - 1, "cell_col": width - 1, "side": "south"}

        return MazeData(
            width=width,
            height=height,
            cell_size=cell_size,
            wall_width=wall_width,
            wall_height=wall_height,
            entrance=entrance,
            exit=exit_,
            grid=grid.tolist(),  # JSON-serialisable
        )

    # Internal helpers

    def _carve_passages(
        self,
        start_row: int,
        start_col: int,
        width: int,
        height: int,
        visited: np.ndarray,
        grid: np.ndarray,
        rng: random.Random,
    ) -> None:
        """Iterative randomised DFS that carves passages through *grid*."""
        stack = [(start_row, start_col)]
        visited[start_row][start_col] = True

        while stack:
            row, col = stack[-1]

            # Collect unvisited neighbours in random order
            neighbours = []
            dirs = list(self._DIRECTIONS)
            rng.shuffle(dirs)
            for dr, dc in dirs:
                nr, nc = row + dr, col + dc
                if 0 <= nr < height and 0 <= nc < width and not visited[nr][nc]:
                    neighbours.append((nr, nc, dr, dc))

            if neighbours:
                # Choose first shuffled unvisited neighbour
                nr, nc, dr, dc = neighbours[0]
                # Carve the wall between current cell and chosen neighbour
                # Wall grid position is midpoint between the two room cells
                grid[2 * row + 1 + dr][2 * col + 1 + dc] = 0
                visited[nr][nc] = True
                stack.append((nr, nc))
            else:
                # Dead end – backtrack
                stack.pop()

    # Convenience: generate + save

    def generate_and_save(
        self,
        path: str,
        width: int,
        height: int,
        **kwargs,
    ) -> MazeData:
        """Generate a maze and immediately save it as JSON."""
        maze = self.generate(width, height, **kwargs)
        maze.save(path)
        return maze
