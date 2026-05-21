"""
SimulationMaze – reconstructs a :class:`~maze_generator.MazeData` as solid
PyBullet bodies so the robot cannot pass through walls.

Wall placement
--------------
The maze grid is a (2*height+1) × (2*width+1) bitmap.  For every cell
``grid[r][c] == 1`` a box collision shape is added at the corresponding world
position.

Physical sizes per grid index:
  even index → wall_width  (shared by MazeData)
  odd  index → cell_size

The maze origin is placed so that (0, 0) world coords coincide with the
north-west corner of the boundary, making it easy to compute entrance /
exit positions.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pybullet as p

from maze_generator.maze_data import MazeData


class SimulationMaze:
    """Loads a :class:`MazeData` and spawns solid wall bodies in PyBullet.

    Parameters
    ----------
    maze_data:
        Logical description of the maze (loaded from JSON or freshly generated).
    origin:
        (x, y) world offset of the maze's north-west corner.
    wall_color:
        RGBA colour of all wall segments.
    """

    def __init__(
        self,
        maze_data: MazeData,
        origin: Tuple[float, float] = (0.0, 0.0),
        wall_color: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 1.0),
    ) -> None:
        self.data = maze_data
        self.origin = origin
        self.wall_color = wall_color
        self._wall_ids: List[int] = []
        self._build()

    # Public helpers

    @classmethod
    def from_file(
        cls,
        path: str,
        origin: Tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ) -> "SimulationMaze":
        """Load maze from a JSON file and spawn it in the simulation."""
        return cls(MazeData.load(path), origin=origin, **kwargs)

    def entrance_world_pos(self) -> Tuple[float, float]:
        """World (x, y) centre of the entrance opening."""
        e = self.data.entrance
        r, c = e["cell_row"], e["cell_col"]
        side  = e["side"]
        return self._opening_center(r, c, side)

    def exit_world_pos(self) -> Tuple[float, float]:
        """World (x, y) centre of the exit opening."""
        ex = self.data.exit
        r, c = ex["cell_row"], ex["cell_col"]
        side  = ex["side"]
        return self._opening_center(r, c, side)

    @property
    def wall_body_ids(self) -> List[int]:
        """PyBullet body ids of all spawned wall segments."""
        return list(self._wall_ids)

    # Internal construction

    def _build(self) -> None:
        """Spawn a static PyBullet box for each wall cell in the grid."""
        md = self.data
        grid_rows = 2 * md.height + 1
        grid_cols = 2 * md.width  + 1

        # Vectorised computation of per-index centres and half-extents
        row_centers  = md.centers_array(grid_rows)          # (grid_rows,)
        col_centers  = md.centers_array(grid_cols)          # (grid_cols,)
        row_half_ext = md.extents_array(grid_rows) / 2.0   # (grid_rows,)
        col_half_ext = md.extents_array(grid_cols) / 2.0   # (grid_cols,)

        wall_half_h = md.wall_height / 2.0
        ox, oy = self.origin

        # Find all wall cells at once with numpy
        grid_np = np.asarray(md.grid, dtype=np.uint8)
        wall_positions = np.argwhere(grid_np == 1)  # shape (N, 2): [[r, c], ...]

        for r, c in wall_positions:
            hx = float(col_half_ext[c])
            hy = float(row_half_ext[r])
            wx = ox + float(col_centers[c])
            wy = oy + float(row_centers[r])

            col_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[hx, hy, wall_half_h],
            )
            vis_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[hx, hy, wall_half_h],
                rgbaColor=list(self.wall_color),
            )

            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=[wx, wy, wall_half_h],
            )
            self._wall_ids.append(body_id)

        print(f"[SimulationMaze] Built {len(self._wall_ids)} wall segments for a "
              f"{md.width}×{md.height} maze.")

    def _opening_center(
        self, cell_row: int, cell_col: int, side: str
    ) -> Tuple[float, float]:
        """Compute world (x, y) of the opening on *side* of the given cell."""
        md = self.data
        ox, oy = self.origin

        # Grid position of the cell centre
        gr = 2 * cell_row + 1
        gc = 2 * cell_col + 1

        # Grid position of the opening (boundary wall grid cell)
        offsets = {"north": (-1, 0), "south": (1, 0),
                   "east":  (0,  1), "west":  (0, -1)}
        dr, dc = offsets[side]
        wr = gr + dr
        wc = gc + dc

        wx = ox + md.physical_center_at(wc)
        wy = oy + md.physical_center_at(wr)
        return wx, wy
