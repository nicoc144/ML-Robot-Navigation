"""
SimLogger – records robot pose, raycast sensor readings, and action labels
during a maze simulation for use as ML training data.

Data collected per sample
--------------------------
* maze_id         – string identifier of the maze
* x, y, theta     – robot pose (world frame; theta = yaw in radians)
* ray_000..ray_N  – raycast distances for each ray (metres, up to ray_length)
* action          – discrete label: "forward" | "left" | "right" |
                    "backward" | "stop"

Output structure
----------------
  <output_dir>/
    <maze_id>/
      <planner>/
        <maze_id>_<planner>_<YYYYMMDD_HHMMSS>.csv

Usage
-----
  logger = SimLogger(
      maze_id="maze_10x10_5",
      planner_name="astar",
      num_rays=36,
      sample_hz=100.0,
      output_dir="/path/to/data/logs",
  )

  # inside simulation step callback:
  logger.log(robot, raycast_readings, action)

  # when done:
  logger.close()
"""

from __future__ import annotations

import csv
import math
import os
import time
from typing import List, Optional

import pybullet as p


# ---------------------------------------------------------------------------
# Action constants – import these from SimLogger for consistency across files
# ---------------------------------------------------------------------------

ACTION_FORWARD  = "forward"
ACTION_LEFT     = "left"
ACTION_RIGHT    = "right"
ACTION_BACKWARD = "backward"
ACTION_STOP     = "stop"


class SimLogger:
    """Logs simulation data to a per-maze/per-planner CSV file.

    Parameters
    ----------
    maze_id:
        Short identifier for the maze (e.g. ``"maze_10x10_5"``).  Becomes
        part of the file path and is stored in every row.
    planner_name:
        Name of the active planner (e.g. ``"astar"``, ``"bfs"``,
        ``"keyboard"``). Becomes part of the file path.
    num_rays:
        Number of rays in the RaycastSensor.  Determines how many
        ``ray_NNN`` columns the CSV will contain.
    sample_hz:
        Target logging frequency in Hz (default ``100.0``).  Based on
        wall-clock time, so it is independent of the physics timestep.
    output_dir:
        Root directory under which ``<maze_id>/<planner>/`` sub-folders
        are created.  Defaults to ``data/logs`` relative to this file.
    """

    def __init__(
        self,
        maze_id: str,
        planner_name: str,
        num_rays: int,
        sample_hz: float = 100.0,
        output_dir: Optional[str] = None,
    ) -> None:
        self._planner  = planner_name
        self._num_rays = num_rays
        self._sample_interval = 1.0 / max(sample_hz, 1e-6)  # seconds
        self._samples_written = 0
        self._last_log_time   = 0.0
        self._start_time      = time.monotonic()  # t=0 for timestep column
        self._closed          = False

        # Build output directory and file path
        if output_dir is None:
            # Default: <maze_project>/data/logs/
            base = os.path.join(os.path.dirname(__file__), "..", "data", "logs")
            output_dir = os.path.normpath(base)

        folder = output_dir
        os.makedirs(folder, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename  = f"{maze_id}_{planner_name}_{timestamp}.csv"
        self._csv_path = os.path.join(folder, filename)

        # Column names: timestep + pose + rays + action
        ray_cols = [f"ray_{i:03d}" for i in range(num_rays)]
        self._fieldnames = ["timestep", "x", "y", "theta"] + ray_cols + ["action"]

        self._file   = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self._fieldnames)  # header

        print(
            f"[SimLogger] {maze_id} / {planner_name} — "
            f"{sample_hz} Hz → {self._csv_path}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def csv_path(self) -> str:
        """Absolute path of the output CSV file."""
        return self._csv_path

    @property
    def samples_written(self) -> int:
        """Number of rows written so far."""
        return self._samples_written

    def log(
        self,
        robot,
        raycast_readings: List[float],
        action: str,
    ) -> bool:
        """Record one sample if the sample interval has elapsed.

        Parameters
        ----------
        robot:
            A ``Robot`` instance with ``get_position()`` and
            ``get_orientation()`` methods.
        raycast_readings:
            List of ray distances returned by a ``RaycastSensor``.
            Must have exactly ``num_rays`` elements.
        action:
            One of the action constants: ``"forward"``, ``"left"``,
            ``"right"``, ``"backward"``, ``"stop"``.

        Returns
        -------
        bool
            ``True`` if a row was actually written, ``False`` otherwise
            (rate-limited).
        """
        if self._closed:
            return False

        now = time.monotonic()
        if now - self._last_log_time < self._sample_interval:
            return False
        self._last_log_time = now
        timestep = round(now - self._start_time, 6)

        # Robot pose
        pos = robot.get_position()        # (x, y, z)
        orn = robot.get_orientation()     # quaternion
        x, y = pos[0], pos[1]
        theta = p.getEulerFromQuaternion(orn)[2]  # yaw (radians)

        row = [timestep, x, y, theta] + list(raycast_readings) + [action]
        self._writer.writerow(row)

        self._samples_written += 1
        return True

    def flush(self) -> None:
        """Flush the internal buffer to disk without closing."""
        if not self._closed:
            self._file.flush()

    def close(self) -> None:
        """Flush and close the CSV file.  Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        self._file.flush()
        self._file.close()
        print(
            f"[SimLogger] Closed — {self._samples_written} samples "
            f"→ {self._csv_path}"
        )

    # Allow use as a context manager
    def __enter__(self) -> "SimLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def maze_id_from_path(maze_path: str) -> str:
    """Derive a clean maze identifier from a file path.

    Examples
    --------
    ``"data/mazes/maze_10x10_5.json"``  →  ``"maze_10x10_5"``
    """
    return os.path.splitext(os.path.basename(maze_path))[0]


def action_from_wheel_speeds(left: float, right: float) -> str:
    """Classify a discrete action label from differential-drive wheel speeds.

    Designed for the sign convention used by ``PathController`` and the
    keyboard handler in ``Simulation._handle_keyboard``.

    Classification rules
    --------------------
    * Both speeds near zero → ``"stop"``
    * Both speeds roughly equal and negative → ``"forward"``
      (wheels spin backward to push robot forward in PyBullet rig)
    * Both speeds roughly equal and positive → ``"backward"``
    * Left > Right → ``"left"`` (right wheel faster → turns left)
    * Right > Left → ``"right"``
    """
    EPSILON = 0.5  # speed threshold to detect "near zero"
    diff = left - right
    avg  = (left + right) / 2.0

    if abs(left) < EPSILON and abs(right) < EPSILON:
        return ACTION_STOP
    if abs(diff) < EPSILON * 2:          # both wheels nearly equal speed
        return ACTION_FORWARD if avg < 0 else ACTION_BACKWARD
    return ACTION_LEFT if diff > 0 else ACTION_RIGHT
