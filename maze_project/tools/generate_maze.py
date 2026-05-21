"""
generate_maze.py – CLI tool to create and save maze JSON files.

Usage
-----
    python tools/generate_maze.py                        # 10×10 maze, random seed
    python tools/generate_maze.py --width 6 --height 6  # custom size
    python tools/generate_maze.py --seed 42              # reproducible
    python tools/generate_maze.py --out data/mazes/my_maze.json
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maze_generator import MazeGenerator
from planning import AStarPlanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a random maze and save it as JSON."
    )
    parser.add_argument("--width",       type=int,   default=10)
    parser.add_argument("--height",      type=int,   default=10)
    parser.add_argument("--cell-size",   type=float, default=1.0,
                        dest="cell_size")
    parser.add_argument("--wall-width",  type=float, default=0.15,
                        dest="wall_width")
    parser.add_argument("--wall-height", type=float, default=1.0,
                        dest="wall_height")
    parser.add_argument("--seed",        type=int,   default=None)
    parser.add_argument("--out",         type=str,   default=None,
                        help="Output path (default: data/mazes/maze_WxH_SEED.json)")
    return parser.parse_args()

# Prints the maze ASCII art with the A* path
def print_maze_with_path(maze, path) -> None:
    """
    Print the maze grid with the A* path overlaid.
 
    Legend
    ------
        █  = wall
        *  = path cell
        S  = start (entrance)
        G  = goal  (exit)
           = open cell (not on path)
    """
    # Build a set of logical (row, col) cells that are on the path
    path_set = set(path) if path else set()
    start_cell = path[0]  if path else None
    goal_cell  = path[-1] if path else None
 
    grid = maze.grid
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
 
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            if grid[r][c] != 0:
                # Wall bitmap cell
                row_str += "█"
            else:
                # Check if this is a logical cell (odd row AND odd col in bitmap)
                is_logical_cell = (r % 2 == 1) and (c % 2 == 1)
                if is_logical_cell:
                    cell = (r // 2, c // 2)
                    if cell == start_cell:
                        row_str += "S"
                    elif cell == goal_cell:
                        row_str += "G"
                    elif cell in path_set:
                        row_str += "*"
                    else:
                        row_str += " "
                else:
                    # Passage / corridor cell between logical cells
                    row_str += " "
        print(row_str)

def main() -> None:
    args = parse_args()

    gen = MazeGenerator()
    maze = gen.generate(
        width=args.width,
        height=args.height,
        cell_size=args.cell_size,
        wall_width=args.wall_width,
        wall_height=args.wall_height,
        seed=args.seed,
    )

    if args.out is None:
        seed_str = str(args.seed) if args.seed is not None else "rnd"
        filename  = f"maze_{args.width}x{args.height}_{seed_str}.json"
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_path  = os.path.join(project_root, "data", "mazes", filename)
    else:
        out_path = args.out

    maze.save(out_path)
          
    start = (maze.entrance["cell_row"], maze.entrance["cell_col"])
    goal  = (maze.exit["cell_row"],     maze.exit["cell_col"])
 
    planner = AStarPlanner(maze)
    path    = planner.find_path(start, goal)

    print("\nASCII preview (S=start, G=goal, *=path):")
    print_maze_with_path(maze, path)

    # If there is no valid A* path print none, else print details about the path
    if path is None:
        print("\n[WARNING] A* could not find a path from entrance to exit!")
    else:
        print(f"\nA* path: {len(path)} steps  |  {start} → {goal}")

if __name__ == "__main__":
    main()
