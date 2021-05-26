'''Copyright 2020 Google LLC

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
from __future__ import division
from __future__ import print_function

import sys
import os
import random
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.beta import Beta

from utils_dp import DoublePendulum, calc_double_E, verification

from model import RuleEncoder, DataEncoder, Net, NaiveModel, SharedNet, DataonlyNet, InequalityDual


def main():
  parser = ArgumentParser()
  # train/test hyper parameters
  parser.add_argument('--L1', type=float, default=1.0, help='Pendulum rod length 1')
  parser.add_argument('--L2', type=float, default=1.0, help='Pendulum rod length 2')
  parser.add_argument('--M1', type=float, default=1.0, help='Pendulum mass 1')
  parser.add_argument('--M2', type=float, default=5.0, help='Pendulum mass 2')
  parser.add_argument('--F1', type=float, default=0.001, help='Friction coefficient 1')
  parser.add_argument('--F2', type=float, default=0.001, help='Friction coefficient 2')
  parser.add_argument('--theta1', type=float, default=0.4725, help='Initial theta 1. If None, it will be sampled from [-np.pi/4, np.pi/4]')
  parser.add_argument('--omega1', type=float, default=0.0, help='Initial omega 1. If None, it will be 0.0')
  parser.add_argument('--theta2', type=float, default=0.3449, help='Initial theta 2. If None, it will be sampled from [-np.pi/4, np.pi/4]')
  parser.add_argument('--omega2', type=float, default=0.0, help='Initial omega 2. If None, it will be 0.0')
  parser.add_argument('--delta_t', type=float, default=0.005, help='Delta t used to generate a sequence')
  parser.add_argument('--tmax', type=int, default=3000, help='Max timesteps')
  parser.add_argument('--sampling_step', type=int, default=20, help='Sampling timesteps')
  parser.add_argument('--device', type=str, default='cuda:0')
  parser.add_argument('--batch_size', type=int, default=64, help='default: 64')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--input_dim_encoder', type=int, default=16)
  parser.add_argument('--output_dim_encoder', type=int, default=64)
  parser.add_argument('--hidden_dim_encoder', type=int, default=64)
  parser.add_argument('--hidden_dim_db', type=int, default=64)
  parser.add_argument('--n_layers', type=int, default=2)
  parser.add_argument('--epochnum', type=int, default=10000, help='default: 1000')
  parser.add_argument('--early_stopping_thld', type=int, default=10, help='default: 10')
  parser.add_argument('--valid_freq', type=int, default=5, help='default: 5')
  parser.add_argument('--dual_lr', type=float, default=1.0, help='default: 1.0')

  args = parser.parse_args()
  print(args)

  device = args.device
  seed = args.seed

  # Generate DP sequence
  print("Generate Double-pendulum sequence...")
  L1, L2, M1, M2, F1, F2 = args.L1, args.L2, args.M1, args.M2, args.F1, args.F2
  init_theta1 = round(args.theta1, 4) if args.theta1 is not None else round(np.random.uniform(-np.pi/4, np.pi/4), 4)
  init_omega1 = round(args.omega1, 4) if args.omega1 is not None else 0.0
  init_theta2 = round(args.theta2, 4) if args.theta2 is not None else round(np.random.uniform(-np.pi/4, np.pi/4), 4)
  init_omega2 = round(args.omega2, 4) if args.omega2 is not None else 0.0

  tmax, dt = args.tmax, args.delta_t
  dp = DoublePendulum(L1, L2, M1, M2, init_theta1, init_omega1, init_theta2, init_omega2, F1, F2)
  t, y = dp.generate(tmax=tmax, dt=dt)

  dp_params = {'M1': dp.M1, 'M2': dp.M2, 'L1': dp.L1, 'L2': dp.L2, 'g': dp.g, 'F1': dp.F1, 'F2': dp.F2}
  print('sequence length: {} ({} sec)'.format(len(y), tmax))
  print('dt: {} (sec)\n'.format(dt))

  subsampling = True
  if subsampling:
    # Fine dt for generation and subsample for learning
    sampling_step = args.sampling_step    # sample a row for every the step.
    sampling_dt = dt*sampling_step
    sampling_ind = np.arange(0, t.shape[0] - 1, sampling_step)
    sampling_t = t[sampling_ind]

    input_output_y = np.concatenate((y[:-1], y[1:]), axis=1)    # [[input, output]]
    X = input_output_y[sampling_ind]

  else:
    sampling_dt = dt
    sampling_t = t

    input_output_y = np.concatenate((y[:-1], y[1:]), axis=1)    # [[input, output]]
    X = input_output_y

  X_np = np.array(X)
  print('subsampled sequence length: {} ({} sec)'.format(len(X), tmax))
  print('sampling dt: {} (sec)'.format(sampling_dt))

  # Data preprocessing
  X = torch.tensor(X_np, dtype=torch.float32, device=device)
  num_samples = X.shape[0]
  input_dim = X.shape[1]//2    # (theta1, omega1, theta2, omega2)

  # 60:10:30 split
  train_X, train_y = X[:int(num_samples*0.6), :input_dim], X[:int(num_samples*0.6), input_dim:]
  valid_X, valid_y = X[int(num_samples*0.6):int(num_samples*0.7), :input_dim], X[int(num_samples*0.6):int(num_samples*0.7), input_dim:]
  test_X, test_y = X[int(num_samples*0.7):, :input_dim], X[int(num_samples*0.7):, input_dim:]

  total_train_sample = len(train_X)
  total_valid_sample = len(valid_X)
  total_test_sample = len(test_X)

  batch_size = args.batch_size
  train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
  valid_loader = DataLoader(TensorDataset(valid_X, valid_y), batch_size=valid_X.shape[0])
  test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=test_X.shape[0])

  print("data size: {}/{}/{}".format(len(train_X), len(valid_X), len(test_X)))

  # Start
  lr = 0.001
  dual_lr = args.dual_lr
  seed=42

  print('model_type: {}, lr: {}, dual_lr: {}, seed: {}'
        .format('ldcdl', lr, dual_lr, seed))

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  SKIP = True    # Delta value (x(t+1)-x(t)) prediction if True else absolute value (x(t+1)) prediction
  merge = 'cat'

#   input_dim = 4
  input_dim_encoder = args.input_dim_encoder
  output_dim_encoder = args.output_dim_encoder
  hidden_dim_encoder = args.hidden_dim_encoder
  hidden_dim_db = args.hidden_dim_db
  output_dim = input_dim
  n_layers = args.n_layers

  rule_encoder = RuleEncoder(input_dim, output_dim_encoder, hidden_dim=hidden_dim_encoder)
  data_encoder = DataEncoder(input_dim, output_dim_encoder, hidden_dim=hidden_dim_encoder)
  model = DataonlyNet(input_dim, output_dim, data_encoder, hidden_dim=hidden_dim_db, n_layers=n_layers, skip=SKIP).to(device)
  dual = InequalityDual(1).to(device)

  total_params = sum(p.numel() for p in model.parameters())
  print("total parameters: {}".format(total_params))

  # Dual loss enforces x-y <= 0.
  loss_dual_func = lambda x: torch.mean(x)
  loss_task_func = nn.L1Loss()    # return scalar (reduction=mean)
  l1_func = nn.L1Loss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  dual_optimizer = optim.SGD(dual.parameters(), lr=dual_lr)

  epochnum = args.epochnum
  early_stopping_thld = args.early_stopping_thld
  counter_early_stopping = 1
  valid_freq = args.valid_freq
  saved_filename = 'dp-model_{:.4f}_{:.1f}_{:.4f}_{:.1f}_{}-seed{}.pt' \
                          .format(init_theta1, init_omega1, init_theta2, init_omega2, dual_lr, seed)

  saved_filename =  os.path.join('saved_models', saved_filename)
  print('saved_filename: {}\n'.format(saved_filename))

  # Training
  for epoch in range(1, epochnum+1):
    model.train()
    dual.train()
    train_loss_task = 0
    train_loss_dual = 0
    train_ratio = 0
    for batch_idx, batch_data in enumerate(train_loader):
      batch_train_x = batch_data[0] + 0.01*torch.randn(batch_data[0].shape).to(device)    # Adding noise
      batch_train_y = batch_data[1]

      # Primal optimization
      optimizer.zero_grad()
      alpha = 0.0
      output = model(batch_train_x, alpha=alpha)
      _, _, curr_E = calc_double_E(batch_train_x, **dp_params)    # E(X_t)    Energy of X_t (Current energy)
      _, _, next_E = calc_double_E(batch_train_y, **dp_params)    # E(X_{t+1})    Energy of X_{t+1} (Next energy from ground truth)
      _, _, pred_E = calc_double_E(output, **dp_params)    # E(\hat{X}_t+1)    Energy of \hat{X}_{t+1} (Next energy from prediction)
      loss_task = loss_task_func(output, batch_train_y)    # state prediction
      loss_dual = loss_dual_func(dual(torch.reshape(pred_E-curr_E, (-1, 1))))
      loss_mae = l1_func(output, batch_train_y).item()
      loss = loss_task + loss_dual    # Constrained baseline
      loss.backward()
      optimizer.step()

      # Dual optimization
      dual_optimizer.zero_grad()
      # Can detatch because dual prameters are not produced through energy functions
      dual_loss = -loss_dual_func(dual(torch.reshape(pred_E.detach()-curr_E.detach(), (-1, 1))))
      dual_loss.backward()
      dual_optimizer.step()

      train_loss_task += loss_task * batch_train_x.shape[0] / total_train_sample
      train_loss_dual += loss_dual * batch_train_x.shape[0] / total_train_sample
      train_ratio += verification(curr_E, pred_E, threshold=0.0).item() * batch_train_x.shape[0] / total_train_sample

    # Evaluate on validation set
    if epoch % valid_freq == 0:
      model.eval()
      dual.eval()
      with torch.no_grad():
        val_loss_task = 0
        val_loss_dual = 0
        val_ratio = 0
        for val_x, val_y in valid_loader:
          val_x += 0.01*torch.randn(val_x.shape).to(device)
          output = model(val_x, alpha=0.0)
          _, _, curr_E = calc_double_E(val_x, **dp_params)
          _, _, pred_E = calc_double_E(output, **dp_params)

          val_loss_task += (loss_task_func(output, val_y).item() * val_x.shape[0] / total_valid_sample)
          val_loss_dual += (loss_dual_func(dual(torch.reshape(pred_E-curr_E, (-1, 1)))) * val_x.shape[0] / total_valid_sample)
          val_ratio += (verification(curr_E, pred_E, threshold=0.0).item() * val_x.shape[0] / total_valid_sample)
        print('[Train] Epoch: {} Loss(Task): {:.6f} Loss(Dual): {:.6f}  Ratio(Rule): {:.3f} (alpha: 0.0)'
              .format(epoch, train_loss_task, train_loss_dual, train_ratio))
        print('[Valid] Epoch: {} Loss(Task): {:.6f} Loss(Dual): {:.6f}  Ratio(Rule): {:.3f} (alpha: 0.0)\t best model is updated %%%%'
              .format(epoch, val_loss_task, val_loss_dual, val_ratio))
        torch.save({
          'epoch': epoch,
          'model_state_dict':model.state_dict(),
          'dual_state_dict':dual.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'dual_optimizer_state_dict': dual_optimizer.state_dict(),
          'loss': val_loss_task
        }, saved_filename)

  # Test
  model_eval = model

  model_eval.eval()
  with torch.no_grad():
    test_loss_task = 0
    for test_x, test_y in test_loader:
      output = model_eval(test_x, alpha=0.0)
      test_loss_task += (loss_task_func(output, test_y).item() * test_x.shape[0] / total_test_sample)  # sum up batch loss

  print('\nTest set: Average loss: {:.8f}\n'.format(test_loss_task))

  # Best model
  alphas = [0.0]
  for alpha in alphas:
    model_eval.eval()
    with torch.no_grad():
      test_loss_task, test_ratio = 0, 0
      for test_x, test_y in test_loader:

        output = model_eval(test_x, alpha=0.0)

        test_loss_task += (loss_task_func(output, test_y).item() * test_x.shape[0] / total_test_sample)  # sum up batch loss

        _, _, curr_E = calc_double_E(test_x, **dp_params)
        _, _, next_E = calc_double_E(test_y, **dp_params)
        _, _, pred_E = calc_double_E(output, **dp_params)

        test_ratio += (verification(curr_E, pred_E, threshold=0.0).item() * test_x.shape[0] / total_test_sample)

      print('Test set: Average loss: {:.8f} (alpha:{})'.format(test_loss_task, alpha))
      print("ratio of verified predictions: {:.6f} (alpha:{})".format(test_ratio, alpha))


if __name__ == '__main__':
  main()
