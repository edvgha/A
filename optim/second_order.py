from typing import Callable, List
import sys
import logging
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from lr_scheduler import LearningRate
        

class SecondOrder:
    def __init__(self,
                 init: np.ndarray,
                 obj: Callable[[np.ndarray], float],
                 obj_grad: Callable[[np.ndarray], np.ndarray],
                 obj_inv_hessian: Callable[[np.ndarray], np.ndarray],
                 lr: LearningRate,
                 tol: float,
                 max_steps: int) -> None:
        self.init = init
        self.objective = obj
        self.objective_gradient = obj_grad
        self.objective_inverse_hessian = obj_inv_hessian
        self.lr = lr
        self.tol = tol
        self.max_steps = max_steps

        self.log_lr: List[float] = []
        self.log_objective_newton: List[float] = []
        self.log_params_newton: List[np.ndarray] = []

        result = minimize(self.objective, self.init, method='BFGS')
        self.arg_min = result.x

        # logger
        self.logger = logging.getLogger('FO')
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def apply_newton(self):
        step = 0
        self.params = self.init
        self.log_objective_newton.append(float(self.objective(self.params)))
        self.log_params_newton.append(self.params)

        while step < self.max_steps:
            # learning rate
            lr = self.lr.get()
            self.lr.update(step + 1)
            self.log_lr.append(lr)
            
            # make a step
            gradient = self.objective_gradient(self.params)
            inv_hessian = self.objective_inverse_hessian(self.params)
            self.params = self.params - lr * np.dot(inv_hessian, gradient)
            self.log_params_newton.append(self.params)

            new_obj_val = self.objective(self.params)
            if new_obj_val > self.log_objective_newton[-1]:
                self.logger.warning(f'Newton step: {step} | Old: {self.log_objective_newton[-1]} | New: {new_obj_val}')

            if abs(new_obj_val - self.log_objective_newton[-1]) < self.tol:
                self.logger.warning(f'Early stop: {step} | Obj: {new_obj_val}')
                break
            self.log_objective_newton.append(float(new_obj_val))

            step += 1

    def plot(self):
        theta1 = np.linspace(-1.5, 2.0, 100)
        theta2 = np.linspace(-0.5, 3.0, 100)
        T1, T2 = np.meshgrid(theta1, theta2)
        Z = self.objective(np.array([T1, T2], dtype=np.float64))

        nw_params = np.array(self.log_params_newton)
        
        fig = plt.figure(figsize=(16, 7))

        # 3D surface view
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(T1, T2, Z, cmap='viridis', antialiased=True, alpha=0.3)
        
        # plot all paths in 3D
        ax1.scatter(nw_params[:len(self.log_objective_newton), 0], nw_params[:len(self.log_objective_newton), 1], 
                    self.log_objective_newton, color='red', s=10, label='Newton') # type: ignore
        
        ax1.set_title('Comparison on 3D Surface')
        ax1.legend()

        # contour map
        ax2 = fig.add_subplot(122)
        contours = ax2.contour(T1, T2, Z, levels=40, cmap='viridis', alpha=0.5)
        ax2.clabel(contours, inline=True, fontsize=8)
        
        # draw trajectories on the contour
        ax2.plot(nw_params[:, 0], nw_params[:, 1], 'r.-', alpha=0.7, label='Newton')
        
        # mark start and end points
        ax2.scatter(self.init[0], self.init[1], color='blue', marker='x', s=100, label='Start', zorder=5)
        ax2.scatter(self.arg_min[0], self.arg_min[1], color='green', marker='*', s=150, label='Global Min', zorder=5)

        ax2.set_title('Newton trajectory')
        ax2.set_xlabel(r'$\theta_1$')
        ax2.set_ylabel(r'$\theta_2$')
        ax2.legend()

        plt.tight_layout()
        plt.show()


def initialize():
    init = np.array([-1.2, 2.5], dtype=np.float64)

    obj = lambda x: 0.5 * (x[0]**2 - x[1])**2 + 0.5 * (x[0] - 1)**2

    def obj_grad(params: np.ndarray) -> np.ndarray:
        x0 = 2 * (params[0]**3 - params[0]*params[1]) + params[0] - 1
        x1 = params[1] - params[0]**2
        return np.array([x0, x1], dtype=np.float64)
    
    def obj_inv_hessian(params: np.ndarray) -> np.ndarray:
        use_SDF = True
        h = np.empty((2, 2), dtype=np.float64)
        h[0][0] = 6 * params[0] * params[0] - 2 * params[1] + 1
        h[0][1] = -2 * params[0]
        h[1][0] = -2 * params[0]
        h[1][1] = 1
        if np.any(np.linalg.eigvals(h) < 0):
            # the hessian is ill-conditioned
            if use_SDF:
                # saddle-free newton
                # convert h = VΛV^T to h_sfn = V|Λ|V^T 
                eigenvalues, eigenvectors = np.linalg.eigh(h)
                abs_eigenvalues = np.abs(eigenvalues)
                abs_eigenvalues = np.maximum(abs_eigenvalues, 1E-6)
                h_sfn = eigenvectors @ np.diag(abs_eigenvalues) @ eigenvectors.T
                return np.linalg.inv(h_sfn)
            else:
                # regularize Levenberg-Marquardt
                I = np.eye(h.shape[0])
                # h = h + lambda * I
                lbd = 0.8 # lbd should be reduced as t -> inf
                h_reg = h + lbd * I
                return np.linalg.inv(h_reg)
        return np.linalg.inv(h)
    
    lr = LearningRate(['constant', 0.1])

    tol = 1e-4

    max_steps = 100

    return [init, obj, obj_grad, obj_inv_hessian, lr, tol, max_steps]
    

if __name__ == '__main__':
    params = initialize()

    so = SecondOrder(init=params[0],
                    obj=params[1],
                    obj_grad=params[2],
                    obj_inv_hessian=params[3],
                    lr=params[4],
                    tol=params[5],
                    max_steps=params[6])
    so.apply_newton()
    so.plot()
    

