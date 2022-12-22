from dataclasses import dataclass
import numpy as np
import copy
import pickle
import GPy
from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar, Type, Union
from sklearn.decomposition import PCA


RegressorT = TypeVar("RegressorT", bound="Regressor")


class Regressor(ABC):

    @classmethod
    @abstractmethod
    def fit(cls: Type[RegressorT], X: np.ndarray, Y: np.ndarray, **kwargs) -> RegressorT:
        ...

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        ...



@dataclass(frozen=True)
class NNRegressor(Regressor):
    X: np.ndarray
    Y: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray) -> "NNRegressor":
        return cls(X, Y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        dists = np.sqrt(np.sum((self.X - x) ** 2, axis=1))
        idx = np.argmin(dists)
        return self.Y[idx]


@dataclass(frozen=True)
class StraightRegressor(Regressor):
    n_wp: int

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray) -> "StraightRegressor":
        _, n_wp, _ = Y.shape
        return cls(n_wp)

    def predict(self, x: np.ndarray) -> np.ndarray:
        # assume that x is composed of q_init and q_goal
        q_init, q_goal = x.reshape(2, -1)
        width = (q_goal - q_init) / (self.n_wp - 1)
        traj = np.array([q_init + i * width for i in range(self.n_wp)])
        return traj

    
@dataclass(frozen=True)
class GPRRegressor(Regressor):
    gp: Union[GPy.models.GPRegression, GPy.models.SparseGPRegression]
    pca: Optional[PCA] = None

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray, pca_dim: Optional[int] = None) -> "GPRRegressor":
        n_data, n_input_dim = X.shape
        Y_flatten = Y.reshape(n_data, -1)

        use_pca = pca_dim is not None
        if use_pca:
            pca = PCA(pca_dim)
            pca.fit(Y_flatten)
            Y_flatten = pca.transform(Y_flatten)
        else:
            pca = None
        
        kernel = GPy.kern.RBF(input_dim=n_input_dim, variance=0.1,lengthscale=0.3, ARD=True) + GPy.kern.White(input_dim=n_input_dim)

        if n_data < 200:
            gp = GPy.models.GPRegression(X, Y_flatten, kernel)
            num_restarts = 10
            gp.optimize_restarts(num_restarts=num_restarts)
        else:
            Z = X[:100]
            gp = GPy.models.SparseGPRegression(X, Y_flatten, Z=Z)
            gp.optimize('bfgs')
        return cls(gp, pca)
            
    def predict(self, x: np.ndarray):
        y, cov = self.gp.predict(np.expand_dims(x, axis=0))
        if self.pca is not None:
            y = self.pca.inverse_transform(np.expand_dims(y, axis=0))[0]
        return y
    

# class DP_GLM_Regressor(Regressor):
# 
#     def fit(self,x,y, n_components = 10, n_init = 20 , weight_type = 'dirichlet_process'):
#         import pbdlib as pbd
#         self.x_joint = np.concatenate([x, y], axis=1)
#         self.n_joint = self.x_joint.shape[1]
#         self.n_in = x.shape[1]
#         self.n_out = y.shape[1]
#         self.joint_model = pbd.VBayesianGMM({'n_components':n_components, 'n_init':n_init, 'reg_covar': 0.00006 ** 2,
#      'covariance_prior': 0.00002 ** 2 * np.eye(self.n_joint),'mean_precision_prior':1e-9,'weight_concentration_prior_type':weight_type})
#         self.joint_model.posterior(data=self.x_joint, dp=False, cov=np.eye(self.n_joint))
# 
#     def predict(self,x, return_gmm=True, return_more = False):
#         result = self.joint_model.condition(x, slice(0, self.n_in), slice(self.n_in, self.n_joint),return_gmm = return_gmm) #
#         
#         if return_gmm:
#             if return_more:
#                 return result[0], result[1], result[2] 
#             else:
#                 index = np.argmax(result[0])
#                 return result[1][index], result[2][index]
#         else:
#             return result[0], result[1]
