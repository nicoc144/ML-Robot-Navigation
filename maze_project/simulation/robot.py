"""
Robot – differential-drive robot built entirely from PyBullet primitives.

Geometry
--------
  body     : cylinder, radius=0.20 m, height=0.10 m
  wheels   : cylinder, radius=0.06 m, height=0.02 m  (left + right)
  casters  : sphere,   radius=0.03 m                 (front + rear)

Joint layout (index)
--------------------
  0 → left  wheel  (REVOLUTE)
  1 → right wheel  (REVOLUTE)
  2 → front caster (FIXED)
  3 → rear  caster (FIXED)

Sensors
-------
Attach any :class:`~simulation.sensors.Sensor` with :meth:`add_sensor`.
Call :meth:`get_all_sensor_readings` to collect all sensor data.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pybullet as p
import math

from .sensors import Sensor


class Robot:
    """A differential-drive robot that lives inside a PyBullet simulation.

    Parameters
    ----------
    start_pos:
        (x, y) spawn position in world coordinates.
    start_yaw:
        Initial heading in radians (default: 0 = facing +x).
    """

    # Wheel joint indices
    _LEFT_WHEEL  = 0
    _RIGHT_WHEEL = 1

    # Physical parameters
    _BODY_RADIUS   = 0.20
    _BODY_HEIGHT   = 0.10
    _WHEEL_RADIUS  = 0.06
    _WHEEL_HEIGHT  = 0.02
    _CASTER_RADIUS = 0.03

    def __init__(
        self,
        start_pos: Tuple[float, float] = (0.0, 0.0),
        start_yaw: float = math.radians(270),
    ) -> None:
        self._sensors: List[Sensor] = []
        self._body_id: int = self._build(start_pos, start_yaw)
        self._apply_friction()

    # Properties

    @property
    def body_id(self) -> int:
        """PyBullet body id of the robot's base link."""
        return self._body_id

    @property
    def sensors(self) -> List[Sensor]:
        """Read-only view of the sensors attached to this robot."""
        return list(self._sensors)

    # Sensor management

    def add_sensor(self, sensor: Sensor) -> None:
        """Attach *sensor* to this robot."""
        self._sensors.append(sensor)

    def get_all_sensor_readings(self) -> Dict[str, List[float]]:
        """Return a mapping of sensor class name → readings list."""
        return {
            type(s).__name__: s.sense(self._body_id)
            for s in self._sensors
        }

    # Motion

    def set_wheel_velocity(self, left: float, right: float) -> None:
        """Set target angular velocity for both drive wheels (rad s⁻¹)."""
        p.setJointMotorControl2(
            self._body_id, self._LEFT_WHEEL,
            p.VELOCITY_CONTROL,
            targetVelocity=left,
            force=5,
        )
        p.setJointMotorControl2(
            self._body_id, self._RIGHT_WHEEL,
            p.VELOCITY_CONTROL,
            targetVelocity=right,
            force=5,
        )

    def stop(self) -> None:
        """Immediately stop both wheels."""
        self.set_wheel_velocity(0.0, 0.0)

    # State queries

    def get_position(self) -> Tuple[float, float, float]:
        """Return the (x, y, z) world position of the robot body."""
        pos, _ = p.getBasePositionAndOrientation(self._body_id)
        return pos

    def get_orientation(self):
        """Return the quaternion orientation of the robot body."""
        _, orn = p.getBasePositionAndOrientation(self._body_id)
        return orn

    # Construction helpers

    def _build(self, start_pos: Tuple[float, float], yaw: float) -> int:
        """Create the PyBullet multi-body and return its id."""
        body_col = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=self._BODY_RADIUS, height=self._BODY_HEIGHT
        )
        body_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self._BODY_RADIUS,
            length=self._BODY_HEIGHT,
            rgbaColor=[0.1, 0.4, 0.9, 1],
        )

        wheel_col = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=self._WHEEL_RADIUS, height=self._WHEEL_HEIGHT
        )
        wheel_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self._WHEEL_RADIUS,
            length=self._WHEEL_HEIGHT,
            rgbaColor=[0.05, 0.05, 0.05, 1],
        )

        caster_col = p.createCollisionShape(
            p.GEOM_SPHERE, radius=self._CASTER_RADIUS
        )
        caster_vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self._CASTER_RADIUS,
            rgbaColor=[0.3, 0.3, 0.3, 1],
        )

        orn = p.getQuaternionFromEuler([0, 0, yaw])

        robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=body_col,
            baseVisualShapeIndex=body_vis,
            basePosition=[start_pos[0], start_pos[1], 0.1],
            baseOrientation=orn,
            linkMasses=[0.2, 0.2, 0.05, 0.05],
            linkCollisionShapeIndices=[wheel_col, wheel_col, caster_col, caster_col],
            linkVisualShapeIndices=[wheel_vis, wheel_vis, caster_vis, caster_vis],
            linkPositions=[
                [0, -0.21,  0.00],   # left  wheel
                [0,  0.21,  0.00],   # right wheel
                [ 0.15, 0, -0.03],   # front caster
                [-0.15, 0, -0.03],   # rear  caster
            ],
            linkOrientations=[
                p.getQuaternionFromEuler([1.57, 0, 0]),
                p.getQuaternionFromEuler([1.57, 0, 0]),
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ],
            linkInertialFramePositions=[[0, 0, 0]] * 4,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * 4,
            linkParentIndices=[0, 0, 0, 0],
            linkJointTypes=[
                p.JOINT_REVOLUTE,
                p.JOINT_REVOLUTE,
                p.JOINT_FIXED,
                p.JOINT_FIXED,
            ],
            linkJointAxis=[
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
            ],
        )
        return robot_id

    def _apply_friction(self) -> None:
        # base friction
        p.changeDynamics(self._body_id, -1, lateralFriction=1)

        # wheel friction
        for j in (self._LEFT_WHEEL, self._RIGHT_WHEEL):
            p.changeDynamics(self._body_id, j, lateralFriction=2)

        # casters
        p.changeDynamics(self._body_id, 2, lateralFriction=0.01)
        p.changeDynamics(self._body_id, 3, lateralFriction=0.01)



