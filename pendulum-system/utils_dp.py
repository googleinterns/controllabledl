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

import torch


class DoublePendulum:
  def __init__(self,
               L1,
               L2,
               M1,
               M2,
               theta1,
               omega1,
               theta2,
               omega2,
               F1=0.1,
               F2=0.1):
    """
    L1, L2 : length of two strings
    M1, M2 : mass of two objectives
    F1, F2 : friction coefficients
    theta1, omega1 : initial angular displacement / velocity of object1
    theta2, omega2 : initial angular displacement / velocity of object2
    """
    self.g = 9.81
    self.L1, self.L2 = L1, L2
    self.M1, self.M2 = M1, M2
    self.F1, self.F2 = F1, F2
    self.theta1, self.omega1 = theta1, omega1
    self.theta2, self.omega2 = theta2, omega2

  def get_config(self):
    print('g: {}'.format(self.g))
    print('L1, L2: {}, {}'.format(self.L1, self.L2))
    print('M1, M2: {}, {}'.format(self.M1, self.M2))
    print('F1, F2: {}, {}'.format(self.F1, self.F2))
    print('theta1, omega1: {}, {}'.format(self.theta1, self.omega1))
    print('theta2, omega2: {}, {}'.format(self.theta2, self.omega2))

  def generate(self,
               tmax,
               dt,
               energy_tol=0.01):
    """
    tmax : maximum time
    dt : time point spacing
    """
    t = np.arange(0, tmax+dt, dt)
    # Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
    y0 = np.array([self.theta1, self.omega1, self.theta2, self.omega2])

    # Do the numerical integration of the equations of motion
    y = odeint(self.deriv_double, y0, t)

    # Total energy from the initial conditions
    params = {'M1': self.M1, 'M2': self.M2,
              'L1': self.L1, 'L2': self.L2,
              'g': self.g,
              'F1': self.F1, 'F2': self.F2}
    _, _, E = calc_double_E(y, **params)

    if np.max(E - E[0]) > energy_tol:
      raise ValueError('Maximum energy drift of {} exceeded.'
                       .format(energy_tol))
    elif np.sum(E[1:] - E[:-1] > energy_tol):
      raise ValueError('Energy at t+1 exceeded energy at t.')

    print("Length (L1,L2), Mass (M1,M2), and "
          "Friction (F1,F2) of a string: ({},{}) ({},{}), ({},{})"
          .format(self.L1, self.L2, self.M1, self.M2, self.F1, self.F2))
    print("Initial theta(degree): {:.6f}({:.6f}),{:.6f}({:.6f})"
          .format(y0[0], 180.*y0[0]/np.pi, y0[2], 180.*y0[2]/np.pi))
    print("Initial omega: {:.6f},{:.6f}".format(y0[1], y0[3]))

    self.t = t
    self.y = y    # save trajectory
    self.dt = dt

    return t, y

  def deriv_double(self, y, t):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    g = self.g
    L1, L2 = self.L1, self.L2
    M1, M2 = self.M1, self.M2
    F1, F2 = self.F1, self.F2

    theta1, z1, theta2, z2 = y

    # inverse matrix
    A1, B1 = (M1+M2)*L1, M2*L2*np.cos(theta1-theta2)
    C1 = -M2*L2*z2**2*np.sin(theta1-theta2) - (M1+M2)*g*np.sin(theta1) - F1*L1*z1

    A2, B2 = M2*L1*np.cos(theta1-theta2), M2*L2
    C2 = M2*L1*z1**2*np.sin(theta1-theta2) - M2*g*np.sin(theta2) - F2*L2*z2

    theta1dot = z1
    theta2dot = z2
    M = np.array([[A1, B1], [A2, B2]])
    C = np.array([[C1], [C2]])
    z1dot, z2dot = np.linalg.inv(M).dot(C)

    return theta1dot, z1dot, theta2dot, z2dot

  def make_plot(self, i, ax, trail_secs=1):
    r = 0.05

    # Unpack z and theta as a function of time
    theta1, theta2 = self.y[:, 0], self.y[:, 2]

    # Convert to Cartesian coordinates of the two bob positions.
    x1 = self.L1 * np.sin(theta1)
    y1 = -self.L1 * np.cos(theta1)
    x2 = x1 + self.L2 * np.sin(theta2)
    y2 = y1 - self.L2 * np.cos(theta2)

    # Plot and save an image of the double pendulum
    # configuration for time point i.
    # The pendulum rods.
    ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k')
    # Circles representing the anchor point of rod 1, and bobs 1 and 2.
    c0 = Circle((0, 0), r/2, fc='k', zorder=10)
    c1 = Circle((x1[i], y1[i]), r, fc='b', ec='b', zorder=10)
    c2 = Circle((x2[i], y2[i]), r, fc='r', ec='r', zorder=10)
    ax.add_patch(c0)
    ax.add_patch(c1)
    ax.add_patch(c2)

    # The trail will be divided into ns segments and plotted as a fading line.
    ns = 20
    # This corresponds to max_trail time points.
    max_trail = int(trail_secs / self.dt)
    s = max_trail // ns
    fps = 10
    di = int(1/fps/self.dt)

    for j in range(ns):
      imin = i - (ns-j)*s
      if imin < 0:
        continue
      imax = imin + s + 1
      # The fading looks better if we square the fractional length along the
      # trail.
      alpha = (j/ns)**2
      ax.plot(x2[imin:imax], y2[imin:imax], c='r', solid_capstyle='butt', lw=2, alpha=alpha)

    # Centre the image on the fixed anchor point, and ensure the axes are equal
    ax.set_xlim(-self.L1-self.L2-r, self.L1+self.L2+r)
    ax.set_ylim(-self.L1-self.L2-r, self.L1+self.L2+r)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig('frames/_img{:04d}.png'.format(i//di), dpi=72)
    plt.cla()

def calc_double_E(y, **kwargs):
  """Return the total energy of the system."""
  g = kwargs['g']
  L1, L2 = kwargs['L1'], kwargs['L2']
  M1, M2 = kwargs['M1'], kwargs['M2']

  if len(y.shape) == 1:
    th1, th1d, th2, th2d = y[0], y[1], y[2], y[3]
  elif len(y.shape) == 2:
    th1, th1d, th2, th2d = y[:, 0], y[:, 1], y[:, 2], y[:, 3]

  if isinstance(y, np.ndarray):
    V = -(M1+M2)*L1*g*np.cos(th1) - M2*L2*g*np.cos(th2) + M1*g*L1 + M2*g*(L1+L2)
    T = 0.5*M1*(L1*th1d)**2 + 0.5*M2*((L1*th1d)**2 + (L2*th2d)**2 +
                                      2*L1*L2*th1d*th2d*np.cos(th1-th2))
  elif isinstance(y, torch.Tensor):
    V = -(M1+M2)*L1*g*torch.cos(th1) - M2*L2*g*torch.cos(th2) + M1*g*L1 + M2*g*(L1+L2)
    T = 0.5*M1*torch.square(L1*th1d) + 0.5*M2*(torch.square(L1*th1d) + torch.square(L2*th2d) +
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
    return 1.0*torch.sum(next_E-curr_E < threshold) / curr_E.shape[0]
  else:
    return 1.0*np.sum(next_E-curr_E < threshold) / curr_E.shape[0]

def make_plot(i, ax, dt=0.01, train_secs=1, r=0.05, **kwargs):

  L1, L2 = kwargs['L1'], kwargs['L2']
  # Plot and save an image of the double pendulum configuration for time
  # point i.
  # The pendulum rods.
  ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k')
  # Circles representing the anchor point of rod 1, and bobs 1 and 2.
  c0 = Circle((0, 0), r/2, fc='k', zorder=10)
  c1 = Circle((x1[i], y1[i]), r, fc='b', ec='b', zorder=10)
  c2 = Circle((x2[i], y2[i]), r, fc='r', ec='r', zorder=10)
  ax.add_patch(c0)
  ax.add_patch(c1)
  ax.add_patch(c2)

  # The trail will be divided into ns segments and plotted as a fading line.
  ns = 20
  max_trail = int(trail_secs / dt)
  s = max_trail // ns

  for j in range(ns):
    imin = i - (ns-j)*s
    if imin < 0:
      continue
    imax = imin + s + 1
    # The fading looks better if we square the fractional length along the
    # trail.
    alpha = (j/ns)**2
    ax.plot(x2[imin:imax], y2[imin:imax],
            c='r',
            solid_capstyle='butt',
            lw=2,
            alpha=alpha)

  # Centre the image on the fixed anchor point, and ensure the axes are equal
  ax.set_xlim(-L1-L2-r, L1+L2+r)
  ax.set_ylim(-L1-L2-r, L1+L2+r)
  ax.set_aspect('equal', adjustable='box')
  plt.axis('off')
  plt.savefig('frames/_img{:04d}.png'.format(i//di), dpi=72)
  plt.cla()
