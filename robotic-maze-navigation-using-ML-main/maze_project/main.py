"""
main.py – entry point for the maze robot simulation.

Quick start
-----------
1. Generate a maze:
    python tools/generate_maze.py --width 10 --height 10 --seed 42

2. Run the simulation:
    python main.py --maze data/mazes/my.json              # keyboard control
    python main.py --maze data/mazes/my.json --guided # A* guided navigation (default)
    python main.py --maze data/mazes/my.json --guided --planner bfs
    python main.py --maze data/mazes/my.json --guided --planner dfs
    python main.py --no-maze                         # empty environment

Keyboard controls (PyBullet window)
-------------------------------------
  ↑ / ↓       forward / backward
  ← / →       turn left / right
  Enter        save sensor recordings and stop collection
  Esc          quit
"""

from __future__ import annotations

import os
import sys

import joblib

# Make sure imports work when run as `python main.py` from maze_project/
sys.path.insert(0, os.path.dirname(__file__))

from simulation import (
    Simulation, RaycastSensor,
    load_maze, build_guided_controller, build_autonomous_controller, parse_args, make_logger, PLANNERS,
)

def main() -> None:
    args      = parse_args()
    maze_data = load_maze(args)

    GMM_PATH = os.path.join('data', 'models', 'gmm_model.pkl')
    MLP_PATH = os.path.join('data', 'models', 'mlp_model.pkl')
    PCA_SCALER_ROT_PATH = os.path.join('data', 'models', 'pca_scaler_rot.pkl')
    STANDARD_SCALER_RAW_PATH = os.path.join('data', 'models', 'standard_scaler_raw.pkl')
    MINMAX_SCALER_RAW_PATH = os.path.join('data', 'models', 'minmax_scaler_raw.pkl')


    sim = Simulation(
        maze_data=maze_data,
        render=not args.headless,
        collect_data=not args.no_collect,
    )

    # Derive ray count directly from whatever RaycastSensors are attached.
    _NUM_RAYS = sum(
        s.num_readings for s in sim.robot.sensors
        if isinstance(s, RaycastSensor)
    )

    if args.autonomous and maze_data is not None:

        # Try to gather all of the models and scalers
        try:
            pca_scaler = joblib.load(PCA_SCALER_ROT_PATH)
        except:
            print("Unable to load PCA rot-invariant scaler, have you tried running the unsupervised preprocessing cell?")
        try:
            gmm = joblib.load(GMM_PATH)
        except:
            print("Unable to load GMM, have you tried training the GMM model?")
        try:
            mlp = joblib.load(MLP_PATH)
        except:
            print("Unable to load MLP, have you tried training the MLP model?")
        try:
            standard_scaler = joblib.load(STANDARD_SCALER_RAW_PATH)
        except:
            print("Unable to load standard scaler,  have you tried running the supervised train/validation/test cell?")
        try:
            minmax_scaler = joblib.load(MINMAX_SCALER_RAW_PATH)
        except:
            print("Unable to load minmax scaler,  have you tried running the supervised preprocessing cell?")

        controller = build_autonomous_controller(maze_data, sim, gmm_model=gmm, mlp_model=mlp, pca_scaler_rot=pca_scaler, standard_scaler_raw=standard_scaler, minmax_scaler_raw=minmax_scaler)

        if controller is not None:
            sim.autonomous = True

            def autonomous_step(simulation: Simulation) -> None:
                simulation.collect = False # Not collecting data
                
                if not controller.done:
                    controller.step()
            
            sim.run(step_callback=autonomous_step)

            return
      

    if args.guided and maze_data is not None:
        planner_cls = PLANNERS[args.planner]
        controller  = build_guided_controller(maze_data, sim, planner_cls)

        if controller is not None:
            sim.guided = True
            logger = make_logger(args, args.planner, _NUM_RAYS)

            def guided_step(simulation: Simulation) -> None:
                if not controller.done:
                    controller.step()
                    action = controller.current_action
                else:
                    action = "stop"
                    if simulation._collect:
                        simulation._collect = False
                        simulation.save_training_data()

                # Log the raycast data, etc.
                if logger is not None:
                    readings = simulation.robot.get_all_sensor_readings()
                    raycast  = readings.get("RaycastSensor", [])
                    logger.log(simulation.robot, raycast, action)

            print(f"[main] Guided navigation mode — {planner_cls.__name__}.")

            # Call guided_step every step, (while loop)
            sim.run(step_callback=guided_step)

            if logger is not None:
                logger.close()
            return

    # Default: keyboard control
    logger = make_logger(args, "keyboard", _NUM_RAYS)

    def keyboard_step(simulation: Simulation) -> None:
        if logger is not None:
            readings = simulation.robot.get_all_sensor_readings()
            raycast  = readings.get("RaycastSensor", [])
            logger.log(simulation.robot, raycast, simulation.last_action)

    sim.run(step_callback=keyboard_step)
    if logger is not None:
        logger.close()


if __name__ == "__main__":
    main()

