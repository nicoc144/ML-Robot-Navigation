"""
run_all.py – run headless simulations on every maze for every planner and
collect labelled sensor data for ML training.

Usage
-----
    # all mazes × all planners
    python tools/run_all.py

    # for full domain randomization
    python tools/run_all.py \
        --sensor-noise 0.03 \
        --steering-noise 0.2 \
        --waypoint-jitter 0.05 \
        --random-endpoints 3

    # limit to specific planners
    python tools/run_all.py --planners astar bfs

    # custom maze directory or log-hz
    python tools/run_all.py --maze-dir data/mazes --log-hz 50

Output
------
    data/logs/<maze_id>_<planner>_<timestamp>.csv

Each CSV contains:  maze_id, x, y, theta, ray_000…ray_N, action
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import time
from multiprocessing import Pool, cpu_count

# Allow running from the tools/ directory or from maze_project/
_TOOLS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_TOOLS_DIR)
sys.path.insert(0, _PROJECT_DIR)

from tqdm import tqdm

from maze_generator import MazeData
from simulation import Simulation, RaycastSensor, build_guided_controller
from simulation.utils import PLANNERS, pick_random_endpoints
from logger import SimLogger, maze_id_from_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch headless simulation — collect ML training data."
    )
    parser.add_argument(
        "--maze-dir", default=None, metavar="DIR",
        help="Directory containing maze JSON files "
             "(default: <project>/data/mazes/).",
    )
    parser.add_argument(
        "--planners", nargs="+", choices=list(PLANNERS), default=list(PLANNERS),
        metavar="PLANNER",
        help=f"Planners to run. Choices: {list(PLANNERS)}. Default: all.",
    )
    parser.add_argument(
        "--log-hz", type=float, default=100.0, metavar="HZ",
        help="Sensor logging frequency in Hz (default: 100).",
    )
    parser.add_argument(
        "--log-dir", default=None, metavar="DIR",
        help="Output directory for CSV files "
             "(default: <project>/data/logs/).",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, metavar="SEC",
        help="Per-run wall-clock timeout in seconds (default: 60). "
             "Runs that take longer are stopped and the partial CSV is kept.",
    )

    # domain randomization flags
    noise_group = parser.add_argument_group(
        "perturbation / domain randomization"
    )
    noise_group.add_argument(
        "--sensor-noise", type=float, default=0.0, metavar="STD",
        help="Std-dev of Gaussian noise added to each raycast distance "
            "(metres).  Default 0 = off.  Try 0.02–0.05.",
    )
    noise_group.add_argument(
        "--steering-noise", type=float, default=0.0, metavar="STD",
        help="Std-dev of Gaussian noise added to wheel velocity commands "
             "(rad/s).  Default 0 = off.  Try 0.1–0.3.",
    )
    noise_group.add_argument(
        "--waypoint-jitter", type=float, default=0.0, metavar="STD",
        help="Std-dev of Gaussian offset applied to each waypoint's x,y "
             "(metres).  Default 0 = off.  Try 0.03–0.08.",
    )
    noise_group.add_argument(
        "--random-endpoints", type=int, default=0, metavar="N",
        help="Number of additional random (start, goal) pairs to run per "
             "maze × planner, on top of the default entrance/exit run.  "
             "Default 0 = only use the maze's fixed entrance/exit.",
    )
    noise_group.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for reproducible perturbation and endpoint "
             "sampling.",
    )

    return parser.parse_args()

def _worker(args):
    return run_one(*args)

# ---------------------------------------------------------------------------
# Per-run simulation
# ---------------------------------------------------------------------------

def run_one(
    maze_path: str,
    planner_name: str,
    log_dir: str,
    log_hz: float,
    timeout: float,
    sensor_noise_std: float = 0.0,
    steering_noise_std: float = 0.0,
    waypoint_jitter_std: float = 0.0,
    start_cell=None,
    goal_cell=None,
) -> int:
    """Run one headless simulation for *maze_path* + *planner_name*.

    Returns the number of samples written.
    """
    maze_data   = MazeData.load(maze_path)
    maze_id     = maze_id_from_path(maze_path)
    planner_cls = PLANNERS[planner_name]

    sim = Simulation(
        maze_data=maze_data,
        render=False,          # headless
        collect_data=False,    # we use SimLogger instead
        sensor_noise_std=sensor_noise_std,
    )

    num_rays = sum(
        s.num_readings for s in sim.robot.sensors
        if isinstance(s, RaycastSensor)
    )

    controller = build_guided_controller(maze_data, sim, planner_cls, start=start_cell, goal=goal_cell, 
                                  waypoint_jitter_std=waypoint_jitter_std, steering_noise_std=steering_noise_std)
    if controller is None:
        print(f"  [skip] No path found — {maze_id} / {planner_name}")
        sim.close()
        return 0

    sim.guided = True

    tag = planner_name
    if start_cell is not None:
        tag = (f"{planner_name}"
               f"_r{start_cell[0]}c{start_cell[1]}"
               f"_r{goal_cell[0]}c{goal_cell[1]}")

    logger = SimLogger(
        maze_id=maze_id,
        planner_name=tag,
        num_rays=num_rays,
        sample_hz=log_hz,
        output_dir=log_dir,
    )

    start = time.monotonic()

    with logger:
        while not controller.done:
            if time.monotonic() - start > timeout:
                print(f"  [timeout] {maze_id} / {planner_name} "
                      f"after {timeout:.0f}s — partial data kept.")
                break

            sim.step()
            controller.step()

            readings = sim.robot.get_all_sensor_readings()
            raycast  = readings.get("RaycastSensor", [])
            logger.log(sim.robot, raycast, controller.current_action)

    sim.close()
    return logger.samples_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    maze_dir = args.maze_dir or os.path.join(_PROJECT_DIR, "data", "mazes")
    log_dir  = args.log_dir  or os.path.join(_PROJECT_DIR, "data", "logs")
    os.makedirs(log_dir, exist_ok=True)

    maze_files = sorted(glob.glob(os.path.join(maze_dir, "*.json")))
    if not maze_files:
        sys.exit(f"[run_all] No maze JSON files found in: {maze_dir}")

    rng = random.Random(args.seed)

    jobs_args = []
    for mf in maze_files:
        maze_data = MazeData.load(mf)
        for pl in args.planners:
            planner_cls = PLANNERS[pl]

            jobs_args.append((
                mf, pl, log_dir, args.log_hz, args.timeout,
                args.sensor_noise, args.steering_noise,
                args.waypoint_jitter,
                None, None,  # default endpoints
            ))

            for _ in range(args.random_endpoints):
                endpoints = pick_random_endpoints(
                    maze_data, planner_cls, rng
                )
                if endpoints is not None:
                    s, g = endpoints
                    jobs_args.append((
                        mf, pl, log_dir, args.log_hz, args.timeout,
                        args.sensor_noise, args.steering_noise,
                        args.waypoint_jitter,
                        s, g,
                    ))
 
    n_mazes   = len(maze_files)
    n_total   = len(jobs_args)
    noise_str = (
        f"sensor={args.sensor_noise}  steering={args.steering_noise}  "
        f"jitter={args.waypoint_jitter}  random_ep={args.random_endpoints}"
    )

    print(
        f"[run_all] {len(maze_files)} maze(s) × {len(args.planners)} planner(s) "
        f"= {n_total} run(s)   |  log-hz={args.log_hz}  timeout={args.timeout}s"
    )

    print(f"[run_all] Perturbation: {noise_str}")
    print(f"[run_all] Output → {log_dir}\n")

    total_samples = 0
    failed        = []
    
    # Sequential Version: Uncomment to run without multiprocessing
    # for job in tqdm(jobs_args, unit="run", desc="Simulating"):
    #     maze_id = maze_id_from_path(job[0])
    #     planner_name = job[1]
    #     tqdm.write(f"  {maze_id}  /  {planner_name}")
    #     try:
    #         n = run_one(*job)
    #         total_samples += n
    #     except Exception as exc:
    #         tqdm.write(f"  [ERROR] {maze_id} / {planner_name}: {exc}")
    #         failed.append((maze_id, planner_name))
    # End of Sequential Version

    # Parallel Version
    pool_size = max(1, cpu_count() - 1)

    print(f"[run_all] Using {pool_size} parallel workers\n")

    with Pool(pool_size) as pool:
        results = []

        for i, result in enumerate(
            tqdm(
                pool.imap_unordered(_worker, jobs_args),
                total=len(jobs_args),
                desc="Simulating",
                unit="run"
            )
        ):
            try:
                total_samples += result
            except Exception as exc:
                failed.append(("unknown", "unknown"))
                tqdm.write(f"[ERROR] Worker failed: {exc}")
    # End of Parallel Version

    print(f"\n[run_all] Done — {total_samples} total samples written to {log_dir}")
    if failed:
        print(f"[run_all] {len(failed)} run(s) failed:")
        for mid, pl in failed:
            print(f"  {mid} / {pl}")


if __name__ == "__main__":
    main()