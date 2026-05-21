"""
Sensor abstractions for the maze robot.

Architecture
------------
Sensor (ABC)
└── RaycastSensor   – uniform 360 ° lidar-style scan

Adding a new sensor type
------------------------
1. Subclass ``Sensor``.
2. Implement ``sense(robot_id: int) -> list[float]``.
3. Attach it to a ``Robot`` instance via ``robot.add_sensor(MySensor())``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pybullet as p


class Sensor(ABC):
    """Abstract base class for all robot sensors.

    Any sensor attached to a :class:`~simulation.robot.Robot` must implement
    :meth:`sense`, which receives the PyBullet body id of the robot and
    returns a list of float readings.
    """

    @abstractmethod
    def sense(self, robot_id: int) -> List[float]:
        """Return a list of scalar readings from this sensor.

        Parameters
        ----------
        robot_id:
            PyBullet body id of the robot performing the sensing.
        """

    @property
    @abstractmethod
    def num_readings(self) -> int:
        """Number of scalar values returned by :meth:`sense`."""


class RaycastSensor(Sensor):
    """Full 360 ° uniform lidar implemented with PyBullet ray-test batches.

    Parameters
    ----------
    num_rays:
        How many rays to cast evenly around the robot.
    ray_length:
        Maximum reach of each ray in metres.
    height_offset:
        Additional Z offset above the robot's base origin where rays originate.
    show_rays:
        When ``True`` debug lines are drawn each call (useful for development).
    noise_std:
        Standard Deviation of Gaussian noise applied to each ray distance (in metres). 0.0 (default) means no noise
        A value around 0.02 to 0.05 simulates realistic sensor noise
    """

    def __init__(
        self,
        num_rays: int = 36,
        ray_length: float = 5.0,
        height_offset: float = 0.1,
        show_rays: bool = False,
        noise_std: float = 0.0,
    ) -> None:
        self._num_rays = num_rays
        self.ray_length = ray_length
        self.height_offset = height_offset
        self.show_rays = show_rays
        self._noise_std = noise_std
        # Precompute unit directions once – avoids per-frame trig
        angles = np.linspace(0.0, 2.0 * np.pi, num_rays, endpoint=False)
        self._cos = np.cos(angles)  # shape (num_rays,)
        self._sin = np.sin(angles)  # shape (num_rays,)
        self._debug_ids = [None] * num_rays


    # Sensor interface

    @property
    def num_readings(self) -> int:
        return self._num_rays

    def sense(self, robot_id: int) -> List[float]:
        """Cast rays and return hit distances.

        Each element in the returned list corresponds to one ray.  If the ray
        hits nothing, the value is ``ray_length`` (max distance).
        """
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        rot = np.array(p.getMatrixFromQuaternion(orn))

        # Rotate precomputed unit directions by robot's current yaw (vectorised)
        rdx = rot[0] * self._cos + rot[1] * self._sin  # (num_rays,)
        rdy = rot[3] * self._cos + rot[4] * self._sin  # (num_rays,)

        oz = pos[2] + self.height_offset
        n  = self._num_rays

        ray_froms = np.column_stack([
            np.full(n, pos[0]),
            np.full(n, pos[1]),
            np.full(n, oz),
        ])
        ray_tos = np.column_stack([
            pos[0] + rdx * self.ray_length,
            pos[1] + rdy * self.ray_length,
            np.full(n, oz),
        ])

        results = p.rayTestBatch(ray_froms.tolist(), ray_tos.tolist())

        # Extract hit info with numpy (avoids a Python-level loop for distances)
        obj_ids      = np.array([r[0] for r in results])
        hit_fractions = np.array([r[2] for r in results])
        distances = np.where(obj_ids != -1,
                             hit_fractions * self.ray_length,
                             self.ray_length)
        

        # injecting sensor noise
        if self._noise_std > 0.0:
            noise = np.random.normal(0.0, self._noise_std, size=n)
            distances = np.clip(distances + noise, 0.0, self.ray_length)
            

        if self.show_rays:
            for idx, result in enumerate(results):

                obj_hit, _, _, hit_pos, _ = result

                if obj_hit != -1:
                    end = hit_pos
                    color = [1,0,0]
                else:
                    end = ray_tos[idx]
                    color = [0,1,0]

                start = ray_froms[idx].tolist()
                end   = list(end)

                if self._debug_ids[idx] is None:
                    # First frame: create the line
                    self._debug_ids[idx] = p.addUserDebugLine(start, end, color)
                else:
                    # Later frames: update the line
                    self._debug_ids[idx] = p.addUserDebugLine(
                        start,
                        end,
                        color,
                        replaceItemUniqueId=self._debug_ids[idx]
                    )

        return distances.tolist()
