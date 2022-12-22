import tqdm
from pathlib import Path
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from square.optimizer import OptimizationBasedPlanner
from square.rrt import RRT
from square.trajectory import Trajectory
from square.world import CircleObstacle, SquareWorld
from typing import Tuple, List
from regression import NN_Regressor, Straight_Regressor, GPy_Regressor


def pick_start_goal(world: SquareWorld) -> Tuple[np.ndarray, np.ndarray]:
    start = np.random.rand(2)
    while world.is_colliding(start):
        start = np.random.rand(2)

    goal = np.random.rand(2)
    while world.is_colliding(goal):
        goal = np.random.rand(2)
    return start, goal


def generate_data(n_data: int, world: SquareWorld) -> List[Trajectory]:

    data: List[Trajectory] = []

    with tqdm.tqdm(total=n_data) as pbar:
        while len(data) < n_data:
            start, goal = pick_start_goal(world)

            # first solve by rrt
            rrt = RRT(start, goal, world)
            rrt_traj = rrt.solve()
            solution_found = rrt_traj is not None
            if solution_found:
                planner = OptimizationBasedPlanner(start, goal, world, world.b_min, world.b_max)
                assert rrt_traj is not None

                plan_result = planner.solve(rrt_traj.resample(20))
                if plan_result.success:
                    obstacle_free = np.all([not world.is_colliding(p) for p in plan_result.traj_solution])
                    if obstacle_free:
                        # now add the data
                        data.append(plan_result.traj_solution)
                        pbar.update(1)
    return data


def traj_list_to_XY(traj_list: List[Trajectory]) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    Y = []
    for traj in traj_list:
        start, goal = traj[0], traj[-1]
        x = np.hstack([start, goal])
        y = traj.numpy()
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)


if __name__ == "__main__":
    sdf1 = CircleObstacle(np.array([0.5, 0.6]), 0.2)
    sdf2 = CircleObstacle(np.array([0.2, 0.4]), 0.1)
    sdf3 = CircleObstacle(np.array([0.7, 0.4]), 0.1)
    world = SquareWorld((sdf1, sdf2, sdf3))

    cache_path = Path("/tmp/memmo_traj_list.cache")
    if cache_path.exists():
        with cache_path.open(mode = "rb") as f:
            traj_list = pickle.load(f)
    else:
        traj_list = generate_data(1000, world)
        with cache_path.open(mode = "wb") as f:
            pickle.dump(traj_list, f)
    traj_list = traj_list[:300]
    X, Y = traj_list_to_XY(traj_list)

    #start, goal = pick_start_goal(world)
    start = np.array([0.1, 0.1])
    goal = np.array([0.9, 0.9])
    x_query = np.hstack([start, goal])

    # fit all regressors
    regressor_table = {
            "nn": NN_Regressor.fit(X, Y),
            "std": Straight_Regressor.fit(X, Y),
            "gpr": GPy_Regressor.fit(X, Y),
    }

    # prediction by all regressors
    result_table = {
            key: regr.predict(x_query) 
            for key, regr in regressor_table.items()}

    # visualize problem setting
    fig, ax = world.visualize(with_contourf = False)
    for traj in traj_list:
        traj.visualize((fig, ax), "ro-", lw=0.4, ms=0.4)
    ax.plot(start[0], start[1], "ko")
    ax.plot(goal[0], goal[1], "ko")

    # visualize prediction results
    plot_styles = ["bo-", "ko-", "co-"]
    for regr_name, plot_style in zip(result_table.keys(), plot_styles):
        result_traj = result_table[regr_name]
        traj = Trajectory(list(result_traj.reshape(-1, 2)))
        traj.visualize((fig, ax), plot_style, lw=2.0, ms=2.0, label=regr_name)
    plt.legend(loc="upper left")
    plt.show()
