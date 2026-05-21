# controller.py
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import pybullet as p


class GuidedController:
    """
    Drives a Robot along a list of (x, y) world-coordinate waypoints
    using proportional heading control.

    Parameters
    ----------
    robot :
        Your Robot instance.
    waypoints :
        Ordered list of (x, y) world positions to visit.
    base_speed :
        Forward wheel velocity (rad/s). Start small (~3.0), tune up.
    k_turn :
        Heading correction gain. Higher = snappier turns, too high = wobble.
    waypoint_threshold :
        Distance (m) at which the robot advances to the next waypoint.
    """

    def __init__(
        self,
        robot,
        waypoints: List[Tuple[float, float]],
        base_speed: float = 5.0,
        k_turn: float = 2.0,
        waypoint_threshold: float = 0.15,
        waypoint_jitter_std: float = 0.0,
        steering_noise_std: float = 0.0,
        
    ) -> None:
        self._robot          = robot
        self._waypoints      = waypoints
        self._idx            = 0          # index of current target waypoint
        self._base_speed     = base_speed
        self._k_turn         = k_turn
        self._threshold      = waypoint_threshold
        self._steering_noise = steering_noise_std

        if waypoint_jitter_std > 0.0 and len(waypoints) > 2:
            jittered = [waypoints[0]]  # keep start
            for wx, wy in waypoints[1:-1]:
                jx = wx + np.random.normal(0.0, waypoint_jitter_std)
                jy = wy + np.random.normal(0.0, waypoint_jitter_std)
                jittered.append((jx, jy))
            jittered.append(waypoints[-1])  # keep goal
            self._waypoints = jittered
        else:
            self._waypoints = list(waypoints)

        self._done           = len(waypoints) == 0
        self._current_action = "stop"   # updated every step(); read by SimLogger
        # Uncomment to see the A* Path
        self._draw_static_path()
        self._heading_line_id = -1
        self._target_line_id = -1
        self._text_id = -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def done(self) -> bool:
        """True once the robot has reached the final waypoint."""
        return self._done

    @property
    def current_action(self) -> str:
        """Discrete action label for the current step.

        Returns one of ``"forward"``, ``"left"``, ``"right"``, ``"stop"``.
        Updated every :meth:`step` call.
        """
        return self._current_action

    @property
    def current_waypoint(self) -> Optional[Tuple[float, float]]:
        """The waypoint the robot is currently driving toward."""
        if self._done:
            return None
        return self._waypoints[self._idx]

    def step(self) -> None:
        """
        Call this ONCE per PyBullet simulation step.
        Computes wheel velocities and writes them to the robot.
        """
        if self._done:
            self._current_action = "stop"
            self._robot.stop()
            return

        # Current pose
        pos = self._robot.get_position()          # (x, y, z)
        orn = self._robot.get_orientation()        # quaternion
        robot_x, robot_y = pos[0], pos[1]

        # Get euler yaw in last posiiton of euler (roll, pitch, yaw)
        robot_yaw = p.getEulerFromQuaternion(orn)[2]

        # Vector to current waypoint
        tx, ty = self._waypoints[self._idx]
        dx = tx - robot_x
        dy = ty - robot_y
        distance = math.sqrt(dx * dx + dy * dy)

        # Advance to next waypoint if close enough
        if distance < self._threshold:
            self._idx += 1
            if self._idx >= len(self._waypoints):
                self._done = True
                self._robot.stop()
                return
            # Recalculate toward new waypoint immediately
            tx, ty = self._waypoints[self._idx]
            dx = tx - robot_x  
            dy = ty - robot_y

        # Heading error
        angle_to_target = math.atan2(-dy, -dx) # Negate dy and dx to match the robot's coordinate frame
        heading_error   = angle_to_target - robot_yaw
        heading_error   = math.atan2(
            math.sin(heading_error), math.cos(heading_error)
        )

        # Draw debug visuals to see the current target and heading error (optional, can be expensive if you have many rays or a long path)
        #self._draw_debug(robot_x, robot_y, robot_yaw, tx, ty, heading_error)

        # Differential drive velocities + action label
        if heading_error > math.radians(10):
            # Turn left if heading error is large positive number, slow down
            left_vel  = (self._base_speed / 3) - self._k_turn * 2.0
            right_vel = (self._base_speed / 3) + self._k_turn * 2.0

            self._current_action = "left"

        elif heading_error < -math.radians(10):
            # Turn right if heading error is large negative number, slow down
            left_vel  = (self._base_speed / 3) + self._k_turn * 2.0
            right_vel = (self._base_speed / 3) - self._k_turn * 2.0

            self._current_action = "right"

        else:
            # Within 20 deg of the waypoint, speed up with gentle turning correction
            if heading_error >= 0:
                left_vel  = self._base_speed - self._k_turn * 0.5
                right_vel = self._base_speed + self._k_turn * 0.5
            else:
                left_vel  = self._base_speed + self._k_turn * 0.5
                right_vel = self._base_speed - self._k_turn * 0.5

            self._current_action = "forward"

        if self._steering_noise > 0.0:
            left_vel  += np.random.normal(0.0, self._steering_noise)
            right_vel += np.random.normal(0.0, self._steering_noise)

        self._robot.set_wheel_velocity(left_vel, right_vel)

    def _draw_static_path(self):
        """Draw the path once with minimal debug overhead."""

        z = 0.2
        self._path_ids = []

        for i in range(len(self._waypoints) - 1):
            x1, y1 = self._waypoints[i]
            x2, y2 = self._waypoints[i + 1]

            line_id = p.addUserDebugLine(
                [x1, y1, z],
                [x2, y2, z],
                [0, 0, 1],
                lineWidth=2
            )

            self._path_ids.append(line_id)

    def _draw_debug(self, robot_x, robot_y, robot_yaw, tx, ty, heading_error):
        z = 0.2
        # robot → waypoint line
        self._target_line_id = p.addUserDebugLine(
            [robot_x, robot_y, z],
            [tx, ty, z],
            [1,1,0],
            lineWidth=2,
            replaceItemUniqueId=self._target_line_id
        )

        # compute heading arrow
        arrow_len = 0.4
        ax = robot_x + arrow_len * math.cos(robot_yaw)
        ay = robot_y + arrow_len * math.sin(robot_yaw)

        # robot heading arrow
        self._heading_line_id = p.addUserDebugLine(
            [robot_x, robot_y, z],
            [ax, ay, z],
            [0,1,0],
            lineWidth=3,
            replaceItemUniqueId=self._heading_line_id
        )

        # heading error text
        self._text_id = p.addUserDebugText(
            f"err={math.degrees(heading_error):.1f}",
            [robot_x, robot_y, z+0.4],
            textSize=1.2,
            replaceItemUniqueId=self._text_id
        )

class AutonomousController:
    
    def __init__(
        self,
        robot,
        gmm_model,
        mlp_model,
        pca_scaler_rot,
        standard_scaler_raw,
        minmax_scaler_raw,
        base_speed: float = 7.0,
        k_turn: float = 2.0
        
    ) -> None:
        self._robot                = robot
        self._gmm                  = gmm_model
        self._mlp                  = mlp_model
        self._pca_rot_scaler       = pca_scaler_rot
        self._standard_scaler_raw  = standard_scaler_raw
        self._minmax_scaler_raw    = minmax_scaler_raw
        self._base_speed           = base_speed
        self._k_turn               = k_turn
        self._done                 = False # Change to true when the robot reaches the final cell
        self._current_action       = "stop"

    @property
    def done(self) -> bool:
        """True once the robot has reached the last cell"""
        return self._done

    @property
    def current_action(self) -> str:
        """Discrete action label for the current step.

        Returns one of ``"forward"``, ``"left"``, ``"right"``, ``"stop"``.
        Updated every :meth:`step` call.
        """
        return self._current_action
    
    # Same function from the unsupervised Jupyter notebook, modified to work with a single n value
    def make_rot_invariant(self, rays: np.ndarray, theta: float) -> np.ndarray:
            """Roll each row's ray readings so index-0 always points world-North."""
            if rays.shape[0] != 1:
                print("ERROR: make_rot_invariant() expects 'rays' to be in the format (1, r), not (n, r)")
                return
            n = rays.shape[1] # Num of raycast sensors ex: 36
            angle_per_ray = 2 * np.pi / n
            shifts = np.round(theta / angle_per_ray).astype(int) % n
            out = np.roll(rays, -shifts)

            return out
    
    def step(self) -> None:
        """
        Call this ONCE per PyBullet simulation step.
        Computes wheel velocities and writes them to the robot.
        """
        if self._done:
            self._current_action = "stop"
            self._robot.stop()
            return
        
        ADD_XY_TO_TRAINING = True
        
        # Get current pose
        pos = self._robot.get_position()          # (x, y, z)
        orn = self._robot.get_orientation()        # quaternion
        robot_x, robot_y = pos[0], pos[1]
        robot_yaw = p.getEulerFromQuaternion(orn)[2]

        # Get the raycast sensor data, reformat as a 2D array
        sensor_data = self._robot.get_all_sensor_readings()
        rays_raw = np.array(sensor_data['RaycastSensor']).reshape(1, -1)
        n_rays = rays_raw.size

        # TODO: Possibly add logic to make sure that the number of raycast sensors is equal to the 
        # number of raycast sensors used for training, could cause unintended errors if not fixed

        # Take the raycast data (1, 36) and make it roation invariant
        rays_rot  = self.make_rot_invariant(rays_raw, robot_yaw)

        # Do PCA on the rot invariant data
        rot_pca = self._pca_rot_scaler.transform(rays_rot)

        # Get the result from the GMM model
        # gmm_result is the current cluster (0 - k) of the data in the shape (1,)
        gmm_cluster_result = self._gmm.predict(rot_pca)
        gmm_posterior_result = self._gmm.predict_proba(rot_pca)

        # NOTE: Make sure 'ADD_XY_TO_TRAINING' in the supervised learning preprocessing
        # is set to the same values
        if ADD_XY_TO_TRAINING == True:
            # Get the current x, y values of the robot
            xy = np.array([[robot_x, robot_y]], dtype=np.float32)
            features = np.hstack([rays_raw, xy])
        else:
            features = rays_raw

        # Use the MinMaxScaler and StandardScaler to transform the point, use raw data
        features_minmax = self._minmax_scaler_raw.transform(features)
        features_standard = self._standard_scaler_raw.transform(features_minmax)

        # Get the result from the MLP model and convert into theta
        mlp_result = self._mlp.predict(features_standard)
        theta_mlp = np.arctan2(mlp_result[:, 0], mlp_result[:, 1])
        theta_mlp = theta_mlp[0]

        # Heading error — normalised to [-π, π]
        heading_error   = theta_mlp - robot_yaw
        heading_error   = math.atan2(
            math.sin(heading_error), math.cos(heading_error)
        )

        print(heading_error)

        # Get the posterior result from GMM, posterior result of 1 signifies high confidence
        # in clustering, result of <1 signifies lower confidence
        post_result = max(gmm_posterior_result[0])

        idx_front_sensor = int(n_rays/2)
        start = idx_front_sensor-3
        end = idx_front_sensor+4
        sensor_data_list = sensor_data['RaycastSensor']
        front_sensors = sensor_data_list[start : end]

        min_front = min(front_sensors)

        # NOTE: The + / - values are flipped, the data is trained on atan2(dy, dx)
        # but expects atan2(-dy, -dx), as a workaround we flipped the signs

        if heading_error > math.radians(40) or min_front < 0.5:
            # Turn right if heading error is large positive number
            left_vel  = (self._base_speed / 3) + self._k_turn * 3.0 
            right_vel = (self._base_speed / 3) - self._k_turn * 3.0 

            self._current_action = "left"

        elif heading_error < -math.radians(40) or min_front < 0.5:
            # Turn left if heading error is large negative number
            left_vel  = (self._base_speed / 3) - self._k_turn * 3.0 
            right_vel = (self._base_speed / 3) + self._k_turn * 3.0 

            self._current_action = "right"

        else:
            # Slow down the robot if GMM is unsure
            speed_modifier = 1
            if post_result < 1:
                speed_modifier = 0.5

            # Within 20 deg of the waypoint, speed up with gentle turning correction
            if heading_error >= 0:
                left_vel  = self._base_speed * speed_modifier + self._k_turn
                right_vel = self._base_speed * speed_modifier - self._k_turn 
            else:
                left_vel  = self._base_speed * speed_modifier - self._k_turn
                right_vel = self._base_speed * speed_modifier + self._k_turn

            self._current_action = "forward"

        self._robot.set_wheel_velocity(left_vel, right_vel)
    


def path_to_waypoints(
    cell_path: List[Tuple[int, int]],
    maze_data,
    maze_origin_x: float,
    maze_origin_y: float,
) -> List[Tuple[float, float]]:
    """
    Convert logical cell path to world (x, y) waypoints.

    Parameters
    ----------
    cell_path :
        List of (row, col) from AStarPlanner.find_path()
    cell_size :
        The --cell-size used when generating the maze (default 1.0 m).
    maze_origin_x/y :
        World coords of cell (0, 0) — top-left corner of the maze.
        Check how your maze builder places walls in PyBullet to get this.
    """

    waypoints = []
    for row, col in cell_path:
        # Logical cell (row, col) → grid index (2*row+1, 2*col+1)
        gc = 2 * col
        gr = 2 * row 
        wx = maze_origin_x + maze_data.physical_center_at(gc)
        wy = maze_origin_y + maze_data.physical_center_at(gr)
        waypoints.append((wx, wy))

    return waypoints