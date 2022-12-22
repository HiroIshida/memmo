## Memmory of motion
Source code in `regression.py` implements memmo of Lembono et al. of RA-L(2020) [1]

The source code is primary based on there relevant implementation found in [2, 3]. But heavy modification was made only on structure level: using dataclass and abstract class to unify the interface and so on. And, the detail implementation including hyper-parameter is left intact.

## requirement
install `square`. (Square is dependency-light playground for planning algorithms)
```
git clone https://github.com/HiroIshida/square.git
cd square && pip3 install -e .
```

Also please install `memmo`'s requirement
```
pip3 install scikit-learn GPy tqdm
```
NOTE: numpy >= 1.24.0 is not compatible with GPy (gpy is actually stale library and using expired numpy feature). Thus, if you are using `numpy>=1.24.0`, you must downgrade to `<=1.23`.

## example
While, memmo's primal target is robot trajectory optimization with higher dof (7~) but, planer point trajectory optimization (2dof) is a good toy problem to visualize the behavior of the algorithm. `example.py` does that.

In `example.py`, we first generate bunch of solution trajectories using trajectory optimization trials (rrt solution as a seed). Then fit many different regressors and compare the regression results by plotting. 

Running following command
```python3
python3 example.py
```
yield the following figure

![result](https://user-images.githubusercontent.com/38597814/209147176-3274e0a9-4859-4002-9da7-db5f27dc59b4.png)

The resulting trajectory outputted by each regressors are plotted with different colors. The red lines are precomputed trajectories that are used to fit regressors. Gray circles are obstacles.

It may take a couple of minutes. For faster result production, set `n_data_use` to be a smaller value.


## reference
[1] Lembono, Teguh Santoso, et al. "Memory of motion for warm-starting trajectory optimization." IEEE Robotics and Automation Letters 5.2 (2020): 2594-2601.

[2] https://github.com/teguhSL/memmo_for_trajopt_codes

[3] https://github.com/teguhSL/memmo_for_trajopt_codes/blob/master/notebooks/regression.py
