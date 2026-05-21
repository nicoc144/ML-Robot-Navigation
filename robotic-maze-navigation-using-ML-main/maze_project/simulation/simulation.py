"""
Simulation – top-level orchestrator that owns the PyBullet world, the maze,
and the robot.

Responsibilities
----------------
* Initialise and tear down PyBullet.
* Load (or receive) a :class:`~simulation.maze.SimulationMaze`.
* Spawn and own a :class:`~simulation.robot.Robot`.
* Provide a clean ``step()`` / ``run()`` interface for the control loop.

Keyboard bindings (when running interactively)
----------------------------------------------
  ↑ / ↓       – forward / backward
  ← / →       – turn left / right
  Enter        – save sensor recordings  and quit data collection
  Escape       – quit simulation
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

import numpy as np
import pybullet as p
import pybullet_data

from .robot import Robot
from .maze import SimulationMaze
from .sensors import RaycastSensor
from maze_generator.maze_data import MazeData


class Simulation:
    """PyBullet simulation that houses a maze and a robot.

    Parameters
    ----------
    maze_data:
        Description of the maze to build.  Pass ``None`` to run without a
        maze (open environment).
    render:
        ``True`` → GUI,  ``False`` → DIRECT (headless).
    timestep:
        Physics time step in seconds (default 1/240).
    robot_start:
        (x, y) of the robot's spawn point.  When *None* the robot is placed
        just inside the entrance automatically (requires *maze_data*).
    collect_data:
        When ``True`` sensor readings are recorded and can be saved.
    """

    def __init__(
        self,
        maze_data: Optional[MazeData] = None,
        render: bool = True,
        timestep: float = 1.0 / 240.0,
        robot_start: Optional[tuple] = None,
        collect_data: bool = True,
        sensor_noise_std: float = 0.0,
    ) -> None:
        self._timestep = timestep
        self._collect = collect_data
        self._training_data: List[List[float]] = []
        self.guided  = False
        self.autonomous = False
        self.last_action = "stop"    # updated by _handle_keyboard; read by SimLogger

        # PyBullet setup
        mode = p.GUI if render else p.DIRECT
        self._physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        # Maze
        self._sim_maze: Optional[SimulationMaze] = None
        if maze_data is not None:
            self._sim_maze = SimulationMaze(maze_data)

        # Robot
        if robot_start is None and self._sim_maze is not None:
            # Place robot just inside the entrance
            ex, ey = self._sim_maze.entrance_world_pos()
            robot_start = (ex, ey + maze_data.cell_size * 0.5)
        elif robot_start is None:
            robot_start = (0.0, 0.0)

        self._robot = Robot(start_pos=robot_start)

        # Attach default 36-ray lidar
        self._robot.add_sensor(
            RaycastSensor(
                num_rays=36,
                ray_length=5.0,
                show_rays=True,
                noise_std=sensor_noise_std,
            )
        )

    # Properties

    @property
    def robot(self) -> Robot:
        return self._robot

    @property
    def sim_maze(self) -> Optional[SimulationMaze]:
        return self._sim_maze

    # Main loop

    def run(self, step_callback: Optional[Callable[["Simulation"], None]] = None) -> None:
        """Run the simulation loop until the user presses Escape.

        Parameters
        ----------
        step_callback:
            Optional function called every physics step with ``self`` as the
            argument.  Use it to implement custom control logic.
        """
        print("[Simulation] Running – Press ESC to quit.")
        if not self.guided and not self.autonomous:
            print("[Keyboard Mode]: Use arrow keys to drive.")

        running = True
        while running:
            # Want to run the simulation faster? Change this number.
            # Warning: Higher numbers may be unstable. This is purely for visualization convenience only.
            # 1 = real time
            steps_per_frame = 1

            for _ in range(steps_per_frame):
                p.stepSimulation()

            time.sleep(self._timestep)

            readings = self._robot.get_all_sensor_readings()
            # Concatenate all sensor arrays into one flat numpy array
            flat = np.concatenate([np.asarray(v) for v in readings.values()])

            if self._collect:
                self._training_data.append(flat)

            if step_callback is not None:
                step_callback(self)

            running = self._handle_keyboard()
            time.sleep(self._timestep)

        self.close()

    def step(self) -> dict:
        """Advance the physics by one time step and return sensor readings."""
        p.stepSimulation()
        return self._robot.get_all_sensor_readings()

    def close(self) -> None:
        """Disconnect from PyBullet."""
        try:
            p.disconnect(self._physics_client)
        except Exception:
            pass

    # Data helpers

    def save_training_data(self, path: str = "raycast_data.npy") -> None:
        """Save collected sensor readings as a numpy array."""
        arr = np.stack(self._training_data) if self._training_data else np.empty((0,))
        np.save(path, arr)
        print(f"[Simulation] Saved {len(self._training_data)} samples → {path}")

    # Internal

    def _handle_keyboard(self) -> bool:
        """Process key presses; return False when the loop should exit."""
        keys = p.getKeyboardEvents()

        if not self.guided and not self.autonomous:
            
            forward = 0.0
            turn    = 0.0

            if p.B3G_UP_ARROW    in keys: forward =  10.0
            if p.B3G_DOWN_ARROW  in keys: forward = -10.0
            if p.B3G_LEFT_ARROW  in keys: turn    =   5.0
            if p.B3G_RIGHT_ARROW in keys: turn    =  -5.0

            # Derive discrete action label for logging
            if forward > 0:
                self.last_action = "forward"
            elif forward < 0:
                self.last_action = "backward"
            elif turn > 0:
                self.last_action = "left"
            elif turn < 0:
                self.last_action = "right"
            else:
                self.last_action = "stop"

            left_speed  = -forward - turn
            right_speed = -forward + turn
            self._robot.set_wheel_velocity(left_speed, right_speed)

        # Enter → save data and stop collecting
        # NOTE: Not really needed anymore, look at README to headlessly collect data.
        if p.B3G_RETURN in keys and self._collect:
            self._collect = False
            self.save_training_data()

        # Escape → quit
        if 27 in keys:  # ESC
            return False

        return True
