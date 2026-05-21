## Repository Structure

### Simulation & Core Code

- `maze_project/`: Core simulation, planning, control, and ML code
- `maze_project/main.py`: Entry point — CLI arguments and main simulation loop
- `maze_project/simulation/`: PyBullet simulation layer (robot, maze, sensors, orchestrator)
- `maze_project/maze_generator/`: Procedural maze generation and JSON serialization
- `maze_project/planning/`: Path planning algorithms (A\*, BFS, DFS)
- `maze_project/control/`: Motion control layer (PD waypoint-tracking controller)
- `maze_project/logger/`: Rate-limited CSV data logger for collecting training data
- `maze_project/tools/`: Helper scripts for maze generation and batch data collection
- `maze_project/data/mazes/`: Saved maze JSON files used for simulation and data collection
- `maze_project/README.md`: Detailed documentation of the simulation architecture and usage

### Machine Learning

- `maze_project/learning/`: ML experiment notebooks
- `maze_project/learning/supervised_linear_mlp.ipynb`: Supervised heading regression (linear, ridge, MLP) with maze-grouped train/test splits
- `maze_project/learning/unsupervised_clustering.ipynb`: Unsupervised clustering (K-Means, GMM) with rotation-invariant features and PCA visualization

### Environment Setup

- `conda/`: Conda environment configuration and setup scripts
- `conda/environment.yml`: Dependency definitions (Python 3.11, PyBullet, scikit-learn, etc.)
- `conda/setup.sh`: Setup script for Linux / macOS
- `conda/setup.bat`: Setup script for Windows

### Report

- `docs/images/`: Figures and visualizations referenced in the midterm report