# Maze Robot Project

A PyBullet simulation of a differential-drive robot navigating a procedurally
generated maze. The project is split into four independent layers: maze
generation, physics simulation, path planning, and motion control.

---

## Project Structure

```
maze_project/
├── main.py               # Entry point – CLI args and main loop only
├── data/
│   ├── mazes/            # Saved maze JSON files
│   └── logs/             # Collected training CSVs (git-ignored)
├── tools/
│   ├── generate_maze.py  # Standalone CLI to create & save maze JSON files
│   └── run_all.py        # Batch headless runner – collects data for all mazes
├── maze_generator/       # Maze generation & data layer
│   ├── maze_data.py      # MazeData dataclass + JSON serialisation
│   └── generator.py      # MazeGenerator (randomised DFS algorithm)
├── simulation/           # PyBullet simulation layer
│   ├── sensors.py        # Sensor ABC + RaycastSensor implementation
│   ├── robot.py          # Robot class (geometry, sensors, motion)
│   ├── maze.py           # SimulationMaze – spawns solid wall bodies
│   ├── simulation.py     # Simulation – orchestrates the whole scene
│   └── utils.py          # CLI helpers, load_maze(), build_controller(), make_logger()
├── planning/             # Path planning layer
│   ├── planner.py        # Abstract Planner base class
│   ├── astar.py          # AStarPlanner – shortest path (A*)
│   ├── bfs.py            # BFSPlanner   – shortest path (BFS)
│   └── dfs.py            # DFSPlanner   – any path (DFS)
├── control/              # Motion control layer
│   └── controller.py     # PathController + path_to_waypoints
└── logger/               # Data-collection layer
    └── sim_logger.py     # SimLogger – rate-limited CSV writer
```

---

## Architecture

### 1. Maze Generation Layer (`maze_generator/`)

#### `MazeData`
A plain dataclass that fully describes a maze:

- **Grid bitmap** – `(2·height + 1) × (2·width + 1)` integer array.
  - `1` = wall cell, `0` = open cell.
  - Odd `(row, col)` positions → room cells (always open).
  - Even `(row, col)` positions → wall/passage cells (carved by the generator).
- **Physical dimensions** – `cell_size`, `wall_width`, `wall_height` (all in metres).
- **Entrance / exit** – each stored as `{"cell_row", "cell_col", "side"}` where
  `side` ∈ `{north, south, east, west}`.
- **Serialisation** – `maze.save(path)` / `MazeData.load(path)` round-trip
  through JSON so mazes are portable and reproducible.

#### `MazeGenerator`
Builds perfect mazes using **iterative randomised DFS** (recursive backtracker):

1. Initialise all cells as walled.
2. Start from `(0, 0)`; push onto a stack and mark visited.
3. While the stack is non-empty, pick a random unvisited neighbour, carve
   the wall between them, push the neighbour.
4. If no unvisited neighbours exist, pop (backtrack).

A *perfect* maze has exactly one path between any two cells, so entrance → exit
is **guaranteed solvable by construction** without any extra checks.

- **Entrance** – north boundary wall of cell `(0, 0)`.
- **Exit** – south boundary wall of cell `(height-1, width-1)`.

---

### 2. Simulation Layer (`simulation/`)

#### `Sensor` (ABC)
Abstract base class every sensor must implement:

```python
class Sensor(ABC):
    def sense(self, robot_id: int) -> List[float]: ...
    def num_readings(self) -> int: ...
```

New sensor types (camera, IMU, GPS …) only need to subclass `Sensor` and
implement `sense()`. They attach to a `Robot` with `robot.add_sensor(MySensor())`.

#### `RaycastSensor`
Concrete `Sensor` that fires `num_rays` rays uniformly around 360 °:

- Uses `p.rayTestBatch` for efficiency.
- Returns a list of hit distances (capped at `ray_length` when nothing is hit).
- Optional debug visualisation (`show_rays=True`).

#### `Robot`
Owns the PyBullet multi-body and exposes a clean interface:

| Method | Purpose |
|---|---|
| `add_sensor(s)` | Attach any `Sensor` instance |
| `get_all_sensor_readings()` | Returns `{sensor_class_name: [floats]}` |
| `set_wheel_velocity(l, r)` | Differential drive control (rad s⁻¹) |
| `stop()` | Zero both wheels |
| `get_position()` | World `(x, y, z)` |
| `get_orientation()` | Quaternion |

**Geometry** (built from PyBullet primitives, no URDF needed):

- Body: cylinder, r = 0.20 m, h = 0.10 m
- Drive wheels (×2): cylinder, r = 0.06 m – joint type `REVOLUTE`
- Casters (×2): sphere, r = 0.03 m – joint type `FIXED`, near-zero friction

#### `SimulationMaze`
Reads a `MazeData` and spawns one **static** PyBullet box (`baseMass=0`) for
every `grid[r][c] == 1` cell. Static bodies have full collision, so the robot
cannot drive through walls.

- Box half-extents are derived from `physical_size_at(idx)` so wall and cell
  widths vary independently.
- `entrance_world_pos()` / `exit_world_pos()` return world `(x, y)` coordinates
  of the openings, allowing the robot or planner to target them.
- `from_file(path)` convenience constructor loads JSON and spawns in one call.

#### `Simulation`
Top-level orchestrator:

1. Connects to PyBullet (GUI or DIRECT/headless).
2. Loads the ground plane and sets gravity.
3. Instantiates `SimulationMaze` (if a `MazeData` is provided).
4. Spawns `Robot` just inside the entrance automatically.
5. Attaches a default `RaycastSensor(num_rays=36)`.
6. Runs the physics loop via `run()` or advances one step at a time via `step()`.

Keyboard bindings (interactive mode):

| Key | Action |
|---|---|
| `↑` / `↓` | Forward / backward |
| `←` / `→` | Turn left / right |
| `Enter` | Save sensor recordings and stop collection |
| `Esc` | Quit |

---

### 3. Planning Layer (`planning/`)

All planners share a common interface defined by the abstract base class.

#### `Planner` (ABC)
Every planner must implement a single method:

```python
class Planner(ABC):
    def find_path(self, start: Cell, goal: Cell) -> Optional[List[Cell]]: ...
```

Wall-aware neighbour enumeration (`_neighbours`) and path reconstruction
(`_reconstruct`) are provided by the base class and shared by all subclasses.

#### `AStarPlanner`
Standard A\* search over the **logical cell grid**:

- Heuristic: Manhattan distance.
- Guaranteed shortest path (fewest cells).

#### `BFSPlanner`
Breadth-first search:

- No heuristic — explores level by level.
- Guaranteed shortest path (fewest cells).

#### `DFSPlanner`
Iterative depth-first search:

- Finds *a* path, **not** necessarily the shortest one.
- Useful as a baseline or for generating meandering routes.

All three planners operate on **logical cell space** `(cell_row, cell_col)`, not
the raw `(2n+1)` bitmap. `find_path(start, goal)` returns a `List[Cell]`
including both endpoints, or `None` if no path exists.

---

### 4. Control Layer (`control/`)

#### `PathController`
Drives a `Robot` along a list of `(x, y)` world-coordinate waypoints using
proportional heading control:

| Parameter | Default | Description |
|---|---|---|
| `base_speed` | 3.0 | Forward wheel velocity (rad s⁻¹) |
| `k_turn` | 2.0 | Heading correction gain |
| `waypoint_threshold` | 0.15 | Advance distance (m) |

- `step()` — call once per simulation tick; writes wheel velocities to the robot.
- `done` property — `True` once the final waypoint is reached.
- Draws the planned path as blue debug lines in PyBullet.

#### `path_to_waypoints`
Converts a logical cell path from any planner into world `(x, y)` waypoints,
accounting for maze origin and cell size.

---

## Quick Start

### Generate a maze

```bash
cd maze_project

# Default 10×10, random seed
python tools/generate_maze.py

# Custom size, reproducible seed
python tools/generate_maze.py --width 12 --height 8 --seed 42

# All options
python tools/generate_maze.py --width 6 --height 6 --seed 0 \
                              --cell-size 1.0 --wall-width 0.15 --wall-height 1.0 \
                              --out data/mazes/my_maze.json
```

The tool prints an ASCII preview with the A\* path overlaid and saves a JSON
file to `mazes/`.

### Run the simulation

```bash
# Keyboard control
python main.py --maze data/mazes/my_maze.json

# Guided Navigation mode – A* (default planner)
python main.py --maze data/mazes/my_maze.json --guided

# Guided Navigation mode – choose a different planner
python main.py --maze data/mazes/my_maze.json --guided --planner bfs
python main.py --maze data/mazes/my_maze.json --guided --planner dfs

# Headless (no GUI, useful for data collection / training)
python main.py --maze data/mazes/my_maze.json --headless

# Run the simulation using the trained unsupervised and supervised models
python main.py --maze data/mazes/my_maze.json --autonomous

# Open environment with no maze
python main.py --no-maze
```

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--maze PATH` | — | Maze JSON file to load |
| `--no-maze` | — | Run without a maze (open floor) |
| `--guided` | off | Follow a planned path instead of keyboard |
| `--planner {astar,bfs,dfs}` | `astar` | Planning algorithm (guided navigation mode only) |
| `--headless` | off | Disable PyBullet GUI |
| `--no-collect` | off | Disable legacy numpy sensor collection |
| `--log-hz HZ` | `100` | Sensor logging frequency in Hz |
| `--no-log` | off | Disable CSV data logging entirely |

### Use a planner programmatically

```python
from maze_generator import MazeData
from planning import AStarPlanner, BFSPlanner, DFSPlanner

maze  = MazeData.load("data/mazes/my_maze.json")
start = (maze.entrance["cell_row"], maze.entrance["cell_col"])
goal  = (maze.exit["cell_row"],     maze.exit["cell_col"])

# Swap to any planner with no other changes
planner = BFSPlanner(maze)
path    = planner.find_path(start, goal)
print(path)
```

---

## Extending the Project

### Add a new planner
Subclass `Planner` and implement `find_path()`. The base class provides
`_neighbours()` and `_reconstruct()` for free:

```python
from planning.planner import Planner, Cell
from typing import List, Optional

class MyPlanner(Planner):
    def find_path(self, start: Cell, goal: Cell) -> Optional[List[Cell]]:
        # your search logic using self._neighbours(cell)
        ...
```

Then register it in `simulation/utils.py`:

```python
PLANNERS = {
    "astar": AStarPlanner,
    "bfs":   BFSPlanner,
    "dfs":   DFSPlanner,
    "mine":  MyPlanner,   # ← add here
}
```

And run with `--planner mine`.

### Add a new sensor type
```python
from simulation.sensors import Sensor

class MySensor(Sensor):
    @property
    def num_readings(self) -> int:
        return 1

    def sense(self, robot_id: int) -> list[float]:
        ...
        return [42.0]

robot.add_sensor(MySensor())
```

### Add a custom controller
Pass a `step_callback` to `Simulation.run()`:

```python
def my_controller(sim):
    readings = sim.robot.get_all_sensor_readings()
    left, right = compute_velocities(readings)
    sim.robot.set_wheel_velocity(left, right)

sim.run(step_callback=my_controller)
```

### Change maze parameters
All physical sizes live in `MazeData` and are re-read by `SimulationMaze` at
load time – no code changes needed.
---

## Data Collection

### Overview

The `logger/` layer records robot observations and action labels during any
simulation run. Data is written as CSV files intended for training ML models.
Logging happens at a configurable rate (default **100 Hz**, wall-clock) and is
independent of the physics timestep.

The logger is active by default in both keyboard and guided navigation mode.
Pass `--no-log` to disable it.

### CSV format

Each file contains one row per recorded sample:

| Column | Type | Description |
|---|---|---|
| `timestep` | float | Seconds since the logger started (wall-clock) |
| `x` | float | Robot X position in world frame (m) |
| `y` | float | Robot Y position in world frame (m) |
| `theta` | float | Robot yaw angle (radians) |
| `ray_000` … `ray_N` | float | Raycast distances, one column per ray (m, capped at `ray_length`) |
| `action` | string | Discrete action label: `forward`, `left`, `right`, `backward`, `stop` |

The number of `ray_*` columns matches the `num_rays` of the `RaycastSensor`
attached to the robot (default: **36** rays ⇒ columns `ray_000`–`ray_035`).

### File naming & location

Files are written to `data/logs/` (git-ignored) with the naming convention:

```
data/logs/<maze_id>_<planner>_<YYYYMMDD_HHMMSS>.csv
```

Examples:
```
data/logs/maze_10x10_5_astar_20260316_143022.csv
data/logs/maze_10x10_5_bfs_20260316_143025.csv
data/logs/maze_10x10_57_dfs_20260316_143102.csv
data/logs/maze_5x5_rnd_keyboard_20260316_150300.csv
```

The maze id and planner are embedded in the filename, so no sub-folders are
needed and files can be loaded individually or globbed together.

### Batch data collection

`tools/run_all.py` runs **headless** simulations for every maze in
`data/mazes/` crossed with every planner and saves all CSVs automatically.

```bash
# All mazes × all planners (astar, bfs, dfs) at 100 Hz
python tools/run_all.py

# Only A* at 50 Hz with a 30 s per-run timeout
python tools/run_all.py --planners astar --log-hz 50 --timeout 30

# Custom maze directory and output location
python tools/run_all.py --maze-dir data/mazes --log-dir data/logs
```

#### `run_all.py` CLI reference

| Flag | Default | Description |
|---|---|---|
| `--maze-dir DIR` | `data/mazes/` | Directory of maze JSON files |
| `--planners PLANNER …` | all | Planners to run (`astar`, `bfs`, `dfs`) |
| `--log-hz HZ` | `100` | Logging frequency in Hz |
| `--log-dir DIR` | `data/logs/` | Output directory for CSV files |
| `--timeout SEC` | `60` | Per-run wall-clock timeout; partial CSV is kept |

### Using `SimLogger` directly

```python
from logger import SimLogger

logger = SimLogger(
    maze_id="maze_10x10_5",
    planner_name="astar",
    num_rays=36,          # must match the sensor attached to the robot
    sample_hz=100.0,
    output_dir="data/logs",
)

# inside your simulation loop:
readings = sim.robot.get_all_sensor_readings()
logger.log(sim.robot, readings["RaycastSensor"], action="forward")

logger.close()  # or use it as a context manager (with logger:)
```

`log()` is rate-limited — call it every physics tick and it will only write a
new row once the configured interval has elapsed.