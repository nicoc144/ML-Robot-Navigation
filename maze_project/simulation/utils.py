"""Utility helpers shared between main.py and other entry points."""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, Optional, Type, List, Tuple

from maze_generator import MazeData
from control import GuidedController, AutonomousController, path_to_waypoints
from planning import Planner, AStarPlanner, BFSPlanner, DFSPlanner
from logger import SimLogger, maze_id_from_path

# ---------------------------------------------------------------------------
# Planner registry – maps CLI name → planner class
# ---------------------------------------------------------------------------

PLANNERS: Dict[str, Type[Planner]] = {
    "astar": AStarPlanner,
    "bfs":   BFSPlanner,
    "dfs":   DFSPlanner,
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Maze robot simulation.")

    maze_group = parser.add_mutually_exclusive_group()
    maze_group.add_argument("--maze", type=str, default=None, metavar="PATH",
                            help="Path to a maze JSON file to load.")
    maze_group.add_argument("--no-maze", action="store_true",
                            help="Run without any maze (open floor).")

    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI (useful for training).")
    parser.add_argument("--no-collect", action="store_true",
                        help="Disable sensor data collection.")
    parser.add_argument("--guided", action="store_true",
                        help="Run in guided navigation mode (follows planned path).")
    parser.add_argument("--autonomous", action="store_true", 
                        help ="Use trained models to autonomously navigate through the maze.")
    parser.add_argument("--planner", choices=PLANNERS, default="astar",
                        help="Planning algorithm for guided navigation mode (default: astar).")

    # Logging
    parser.add_argument("--log-hz", type=float, default=100.0, metavar="HZ",
                        help="Sensor logging frequency in Hz (default: 100).")
    parser.add_argument("--no-log", action="store_true",
                        help="Disable data logging entirely.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------


def make_logger(
    args: argparse.Namespace,
    planner_name: str,
    num_rays: int,
) -> Optional[SimLogger]:
    """Return a configured :class:`~logger.SimLogger`, or ``None`` when
    logging is disabled via ``--no-log``."""
    if args.no_log:
        return None
    maze_id = maze_id_from_path(args.maze) if args.maze else "no_maze"
    # data/logs lives at <maze_project>/data/logs/
    log_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", "logs")
    )
    return SimLogger(
        maze_id=maze_id,
        planner_name=planner_name,
        num_rays=num_rays,
        sample_hz=args.log_hz,
        output_dir=log_dir,
    )


def load_maze(args: argparse.Namespace) -> Optional[MazeData]:
    """Return a MazeData from ``--maze PATH``, or ``None`` for ``--no-maze``.

    Exits with a helpful message if neither flag is supplied.
    """
    if args.no_maze:
        return None
    if args.maze is None:
        raise SystemExit(
            "[main] ERROR: Provide a maze file with --maze PATH, "
            "or use --no-maze for an empty environment.\n"
            "       Generate mazes with: python tools/generate_maze.py"
        )
    return MazeData.load(args.maze)

def get_open_cells(maze_data: MazeData) -> List[Tuple[int, int]]:
    """Return all logical (row, col) cells that are open passages.

    A logical cell sits at bitmap position (2*row+1, 2*col+1).
    """
    cells = []
    for r in range(maze_data.height):
        for c in range(maze_data.width):
            br = 2 * r + 1
            bc = 2 * c + 1
            if maze_data.grid[br][bc] == 0:
                cells.append((r, c))
    return cells


def pick_random_endpoints(
    maze_data: MazeData,
    planner_cls: Type[Planner],
    rng: random.Random,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Pick a random (start, goal) pair that is reachable.

    Tries up to 50 random pairs before giving up and returning ``None``.
    """
    open_cells = get_open_cells(maze_data)
    if len(open_cells) < 2:
        return None

    for _ in range(50):
        start, goal = rng.sample(open_cells, 2)
        path = planner_cls(maze_data).find_path(start, goal)
        if path is not None and len(path) >= 3:
            return start, goal
    return None


def build_guided_controller(
    maze_data: MazeData,
    sim,
    planner_cls: Type[Planner],
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    waypoint_jitter_std: float = 0.0,
    steering_noise_std: float = 0.0,
) -> Optional[GuidedController]:
    """Plan a path with *planner_cls* and return a PathController, or ``None``.

    Parameters
    ----------
    start, goal:
        Logical (row, col) cells.  When ``None`` the maze's entrance / exit
        are used (original behaviour).
    waypoint_jitter_std:
        Passed through to :class:`PathController`.
    steering_noise_std:
        Passed through to :class:`PathController`.
    """
    if start is None:
        start = (maze_data.entrance["cell_row"], maze_data.entrance["cell_col"])
    if goal is None:
        goal  = (maze_data.exit["cell_row"],     maze_data.exit["cell_col"])

    cell_path = planner_cls(maze_data).find_path(start, goal)

    if cell_path is None:
        print(f"[main] WARNING: {planner_cls.__name__} found no path — "
              "falling back to keyboard control.")
        return None

    print(f"[main] {planner_cls.__name__}: {len(cell_path)} steps  |  {start} → {goal}")

    ex, ey = sim.sim_maze.entrance_world_pos()
    waypoints = path_to_waypoints(
        cell_path,
        maze_data=maze_data,
        maze_origin_x=ex,
        maze_origin_y=ey + maze_data.cell_size * 0.5,
    )
    return GuidedController(
        sim.robot,
        waypoints,
        waypoint_jitter_std=waypoint_jitter_std,
        steering_noise_std=steering_noise_std,
    )

def build_autonomous_controller(
        maze_data: MazeData,
        sim,
        gmm_model,
        mlp_model,
        pca_scaler_rot,
        standard_scaler_raw,
        minmax_scaler_raw,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
    ) -> Optional[AutonomousController]:

    if start is None:
        start = (maze_data.entrance["cell_row"], maze_data.entrance["cell_col"])
    if goal is None:
        goal  = (maze_data.exit["cell_row"],     maze_data.exit["cell_col"])

    return AutonomousController(
        sim.robot,
        gmm_model=gmm_model,
        mlp_model=mlp_model,
        pca_scaler_rot=pca_scaler_rot,
        standard_scaler_raw=standard_scaler_raw,
        minmax_scaler_raw=minmax_scaler_raw
    )
