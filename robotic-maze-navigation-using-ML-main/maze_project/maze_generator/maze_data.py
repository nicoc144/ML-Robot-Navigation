"""
MazeData – serialisable description of a maze layout.

Grid encoding
-------------
The maze is stored as a (2*height + 1) × (2*width + 1) bitmap where

    grid[r][c] == 1  → wall
    grid[r][c] == 0  → passage / open space

Odd  (r, c) positions  → room cells        (always 0 after generation)
Even (r, c) positions  → potential walls   (0 if passage was carved)

Physical layout
---------------
Each grid index maps to a physical extent:
    even index  → wall_width (metres)
    odd  index  → cell_size  (metres)

So the physical centre of grid column c is the sum of all extents to the
left plus half the extent at c, and identically for rows.

Entrance / exit
---------------
Stored as a dict  {"cell_row": int, "cell_col": int, "side": str}
where "side" is one of {"north", "south", "east", "west"}.
The corresponding boundary wall cell in the grid is opened (set to 0).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import numpy as np


@dataclass
class MazeData:
    # Logical dimensions
    width:  int   # number of cells horizontally
    height: int   # number of cells vertically

    # Physical dimensions (metres)
    cell_size:    float  # open space inside one cell
    wall_width:   float  # thickness of a wall segment
    wall_height:  float  # Z height of every wall (used in simulation)

    # Special cells
    entrance: Dict[str, Any]  # {"cell_row", "cell_col", "side"}
    exit:     Dict[str, Any]  # {"cell_row", "cell_col", "side"}

    # Grid bitmap (2*height+1) × (2*width+1)
    grid: List[List[int]]

    # Helpers – physical sizing

    def physical_size_at(self, idx: int) -> float:
        """Return the physical extent (m) for grid index *idx*.

        Even idx → wall_width, Odd idx → cell_size.
        """
        return self.wall_width if idx % 2 == 0 else self.cell_size

    def extents_array(self, n: int) -> np.ndarray:
        """Return a float64 array of physical extents for grid indices 0..n-1.

        Even indices → wall_width, odd → cell_size.
        """
        idx = np.arange(n)
        return np.where(idx % 2 == 0, self.wall_width, self.cell_size)

    def physical_center_at(self, idx: int) -> float:
        """Return the physical centre coordinate (m) for grid index *idx*.

        Measured from the outer left/top edge of the maze boundary.
        """
        ext = self.extents_array(idx + 1)
        return float(ext[:idx].sum() + ext[idx] / 2.0)

    def centers_array(self, n: int) -> np.ndarray:
        """Return a float64 array of physical centres for grid indices 0..n-1."""
        ext = self.extents_array(n)
        # cumsum gives right edges; subtract half-extent for centres
        return np.cumsum(ext) - ext / 2.0

    def total_width_m(self) -> float:
        """Total physical width of the maze (m)."""
        return float(self.extents_array(2 * self.width + 1).sum())

    def total_height_m(self) -> float:
        """Total physical height of the maze (m)."""
        return float(self.extents_array(2 * self.height + 1).sum())

    # Serialisation

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MazeData":
        return cls(**data)

    def save(self, path: str) -> None:
        """Persist the maze to a JSON file, creating directories as needed."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[MazeData] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MazeData":
        """Load a maze from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        maze = cls.from_dict(data)
        print(f"[MazeData] Loaded {maze.width}×{maze.height} maze from {path}")
        return maze
