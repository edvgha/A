from typing import Callable, List
import sys
import logging
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from lr_scheduler import LearningRate

class FirstOrder:
    def __init__(self,
                 init: np.ndarray,
                 obj: Callable[[np.ndarray], float],
                 obj_grad: Callable[[np.ndarray], np.ndarray], 
                 lr: LearningRate,
                 tol: float,
                 max_steps: int) -> None:
        self.init = init
        self.objective = obj
        self.objective_gradient = obj_grad
        self.lr = lr
        self.tol = tol
        self.max_steps = max_steps

        self.log_lr: List[float] = []
        # gradient descent
        self.log_objective_gd: List[float] = []
        self.log_params_gd: List[np.ndarray] = []
        # momentum
        self.log_objective_mo: List[float] = []
        self.log_params_mo: List[np.ndarray] = []
        # nesterov
        self.log_objective_ne: List[float] = []
        self.log_params_ne: List[np.ndarray] = []

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

    def apply_nesterov(self, beta: float = .9):
        step, momentum = 0, .0
        self.params = self.init
        self.log_objective_ne.append(float(self.objective(self.params)))
        self.log_params_ne.append(self.params)

        while step < self.max_steps:
            # learning rate
            lr = self.lr.get()
            self.lr.update(step + 1)
            self.log_lr.append(lr)
            
            # make a step
            gradient = self.objective_gradient(self.params)
            momentum = beta * momentum - lr * gradient
            self.params = self.params + momentum
            self.log_params_ne.append(self.params)

            new_obj_val = self.objective(self.params)
            if new_obj_val > self.log_objective_ne[-1]:
                self.logger.warning(f'Nesterov step: {step} | Old: {self.log_objective_ne[-1]} | New: {new_obj_val}')

            if abs(new_obj_val - self.log_objective_ne[-1]) < self.tol:
                break
            self.log_objective_ne.append(float(new_obj_val))

            step += 1

    def apply_momentum(self, beta: float = .9):
        step, momentum = 0, .0
        self.params = self.init
        self.log_objective_mo.append(float(self.objective(self.params)))
        self.log_params_mo.append(self.params)

        while step < self.max_steps:
            # learning rate
            lr = self.lr.get()
            self.lr.update(step + 1)
            self.log_lr.append(lr)
            
            # make a step
            gradient = self.objective_gradient(self.params)
            momentum = beta * momentum + gradient
            self.params = self.params - lr * momentum
            self.log_params_mo.append(self.params)

            new_obj_val = self.objective(self.params)
            if new_obj_val > self.log_objective_mo[-1]:
                self.logger.warning(f'Momentum step: {step} | Old: {self.log_objective_mo[-1]} | New: {new_obj_val}')

            if abs(new_obj_val - self.log_objective_mo[-1]) < self.tol:
                break
            self.log_objective_mo.append(float(new_obj_val))

            step += 1

    def apply_gd(self):
        step = 0
        self.params = self.init
        self.log_objective_gd.append(float(self.objective(self.params)))
        self.log_params_gd.append(self.params)

        while step < self.max_steps:
            # learning rate
            lr = self.lr.get()
            self.lr.update(step + 1)
            self.log_lr.append(lr)
            
            # make a step
            gradient = self.objective_gradient(self.params)
            self.params = self.params - lr * gradient
            self.log_params_gd.append(self.params)

            new_obj_val = self.objective(self.params)
            if new_obj_val > self.log_objective_gd[-1]:
                self.logger.warning(f'GD step: {step} | Old: {self.log_objective_gd[-1]} | New: {new_obj_val}')

            if abs(new_obj_val - self.log_objective_gd[-1]) < self.tol:
                break
            self.log_objective_gd.append(float(new_obj_val))

            step += 1
    
    def plot(self):
        theta1 = np.linspace(-1.5, 2.0, 100)
        theta2 = np.linspace(-0.5, 3.0, 100)
        T1, T2 = np.meshgrid(theta1, theta2)
        Z = self.objective(np.array([T1, T2], dtype=np.float64))

        gd_params = np.array(self.log_params_gd)
        mo_params = np.array(self.log_params_mo)
        ne_params = np.array(self.log_params_ne)
        
        fig = plt.figure(figsize=(16, 7))

        # 3D surface view
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(T1, T2, Z, cmap='viridis', antialiased=True, alpha=0.3)
        
        # plot all paths in 3D
        ax1.scatter(gd_params[:len(self.log_objective_gd), 0], gd_params[:len(self.log_objective_gd), 1], 
                    self.log_objective_gd, color='red', s=10, label='GD') # type: ignore
        ax1.scatter(mo_params[:len(self.log_objective_mo), 0], mo_params[:len(self.log_objective_mo), 1], 
                    self.log_objective_mo, color='orange', s=10, label='Momentum') # type: ignore
        ax1.scatter(ne_params[:len(self.log_objective_ne), 0], ne_params[:len(self.log_objective_ne), 1], 
                    self.log_objective_ne, color='cyan', s=10, label='Nesterov') # type: ignore
        
        ax1.set_title('Comparison on 3D Surface')
        ax1.legend()

        # contour map
        ax2 = fig.add_subplot(122)
        contours = ax2.contour(T1, T2, Z, levels=40, cmap='viridis', alpha=0.5)
        ax2.clabel(contours, inline=True, fontsize=8)
        
        # draw trajectories on the contour
        ax2.plot(gd_params[:, 0], gd_params[:, 1], 'r.-', alpha=0.7, label='GD')
        ax2.plot(mo_params[:, 0], mo_params[:, 1], '.-', color='orange', alpha=0.7, label='Momentum')
        ax2.plot(ne_params[:, 0], ne_params[:, 1], '.-', color='cyan', alpha=0.7, label='Nesterov')
        
        # mark start and end points
        ax2.scatter(self.init[0], self.init[1], color='blue', marker='x', s=100, label='Start', zorder=5)
        ax2.scatter(self.arg_min[0], self.arg_min[1], color='green', marker='*', s=150, label='Global Min', zorder=5)

        ax2.set_title('GD vs. Momentum vs. Nesterov trajectories')
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
    
    lr = LearningRate(['constant', 0.1])

    tol = 1e-4

    max_steps = 100

    return [init, obj, obj_grad, lr, tol, max_steps]
    

if __name__ == '__main__':
    params = initialize()

    fo = FirstOrder(init=params[0],
                    obj=params[1],
                    obj_grad=params[2],
                    lr=params[3],
                    tol=params[4],
                    max_steps=params[5])
    fo.apply_gd()
    fo.apply_momentum(beta=.58)
    fo.apply_nesterov(beta=.45)
    fo.plot()
    

