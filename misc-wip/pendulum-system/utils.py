'''
Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import sys

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# import tensorflow as tf
import torch


class DoublePendulum:
    def __init__(self, L1, L2, M1, M2, theta1, omega1, theta2, omega2):
        """
        L1, L2 : length of two strings
        M1, M2 : mass of two objectives
        theta1, omega1 : initial angular displacement / velocity of object1
        theta2, omega2 : initial angular displacement / velocity of object2
        """
        self.g = 9.81
        self.L1, self.L2 = L1, L2
        self.M1, self.M2 = M1, M2
        self.theta1, self.omega1 = theta1, omega1
        self.theta2, self.omega2 = theta2, omega2
        
    def get_config(self):
        print('g: {}'.format(self.g))
        print('L1, L2: {}, {}'.format(self.L1, self.L2))
        print('M1, M2: {}, {}'.format(self.M1, self.M2))
        print('theta1, omega1: {}, {}'.format(self.theta1, self.omega1))
        print('theta2, omega2: {}, {}'.format(self.theta2, self.omega2))
        
    def generate(self, tmax, dt, energy_tol=0.01):
        """
        tmax : maximum time
        dt : time point spacing
        """
        t = np.arange(0, tmax+dt, dt)
        # Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
        y0 = np.array([self.theta1, self.omega1, self.theta2, self.omega2])
        
        # Do the numerical integration of the equations of motion
        y = odeint(self.deriv_double, y0, t)

        # Check that the calculation conserves total energy to within some tolerance.
        # Total energy from the initial conditions
        params = {'M1': self.M1, 'M2': self.M2, 'L1': self.L1, 'L2': self.L2, 'g': self.g}
        _, _, E = calc_double_E(y0, **params)
        if np.max(np.abs(calc_double_E(y, **params)[2] - E)) > energy_tol:
            raise ValueError('Maximum energy drift of {} exceeded.'.format(energy_tol))

#         # Calculate energy
#         V, T, E = calc_double_E(y, **params)

        print("Length (L1,L2) and Mass (M1,M2) of a string: ({},{}) ({},{})".format(self.L1, self.L2, self.M1, self.M2))
        print("Initial theta(degree): {:.6f}({:.6f}),{:.6f}({:.6f})".format(y0[0], 180.*y0[0]/np.pi, y0[2], 180.*y0[2]/np.pi))
        print("Initial omega: {:.6f},{:.6f}".format(y0[1], y0[3]))
        
        return t, y

    def deriv_double(self, y, t):
        """Return the first derivatives of y = theta1, z1, theta2, z2."""
        g = self.g
        L1, L2 = self.L1, self.L2
        M1, M2 = self.M1, self.M2
        
        theta1, z1, theta2, z2 = y

        c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

        theta1dot = z1
        z1dot = (M2*g*np.sin(theta2)*c - M2*s*(L1*z1**2*c + L2*z2**2) -
                 (M1+M2)*g*np.sin(theta1)) / L1 / (M1 + M2*s**2)
        theta2dot = z2
        z2dot = ((M1+M2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
                 M2*L2*z2**2*s*c) / L2 / (M1 + M2*s**2)
        return theta1dot, z1dot, theta2dot, z2dot
    
    
def calc_double_E(y, **kwargs):
    """Return the total energy of the system."""
    g = kwargs['g']
    L1, L2 = kwargs['L1'], kwargs['L2']
    M1, M2 = kwargs['M1'], kwargs['M2']
    
    if len(y.shape) == 1:
        th1, th1d, th2, th2d = y[0], y[1], y[2], y[3]
    elif len(y.shape) == 2:
        th1, th1d, th2, th2d = y[:,0], y[:,1], y[:,2], y[:,3]
    
    if isinstance(y, np.ndarray):
        V = -(M1+M2)*L1*g*np.cos(th1) - M2*L2*g*np.cos(th2) + M1*g*L1 + M2*g*(L1+L2)
        T = 0.5*M1*(L1*th1d)**2 + 0.5*M2*((L1*th1d)**2 + (L2*th2d)**2 +
                2*L1*L2*th1d*th2d*np.cos(th1-th2))
#     elif isinstance(y, tf.Tensor):
#         V = -(M1+M2)*L1*g*tf.math.cos(th1) - M2*L2*g*tf.math.cos(th2) + M1*g*L1 + M2*g*(L1+L2)
#         T = 0.5*M1*(L1*th1d)**2 + 0.5*M2*((L1*th1d)**2 + (L2*th2d)**2 +
#                 2*L1*L2*th1d*th2d*tf.math.cos(th1-th2))
    elif isinstance(y, torch.Tensor):
        V = -(M1+M2)*L1*g*torch.cos(th1) - M2*L2*g*torch.cos(th2) + M1*g*L1 + M2*g*(L1+L2)
        T = 0.5*M1*(L1*th1d)**2 + 0.5*M2*((L1*th1d)**2 + (L2*th2d)**2 +
                2*L1*L2*th1d*th2d*torch.cos(th1-th2))
    else:
        raise TypeError("type of y is :{}. It should be numpy.ndarray or torch.Tensor".format(type(y)))

    return (V, T, T + V)


def calc_single_E(y, **kwargs):
    """Return the total energy of the system."""
    g = kwargs['g']
    L = kwargs['L']
    M = kwargs['M']
    
    if isinstance(y, np.ndarray):
        theta, theta_dot = y.T
        V = M*g*L*(1-np.cos(theta))
        T = 0.5*M*(L*theta_dot)**2
    elif isinstance(y, tf.Tensor):
        theta, theta_dot = tf.transpose(y)
        V = M*g*L*(1-torch.cos(theta))
        T = 0.5*M*(L*theta_dot)**2
    elif isinstance(y, torch.Tensor):
        theta, theta_dot = y.T
        V = M*g*L*(1-torch.cos(theta))
        T = 0.5*M*(L*theta_dot)**2
    else:
        raise TypeError("type of y is :{}. It should be numpy.ndarray or torch.Tensor".format(type(y)))
    
    return (V, T, T + V)



def verification(curr_E, next_E, threshold=0.1):
    '''
    return the ratio of qualified samples.
    '''
    if isinstance(curr_E, torch.Tensor):
        return 1.0*torch.sum(next_E-curr_E <= threshold) / curr_E.shape[0]
    else:
        return 1.0*np.sum(next_E-curr_E <= threshold) / curr_E.shape[0]