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

import os
import random
from copy import deepcopy
from argparse import ArgumentParser

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.beta import Beta
from torch.utils.data import DataLoader, TensorDataset

from model import RuleEncoder, DataEncoder, M5Net, InequalityDual

import utils
from utils_learning import verification, get_perturbed_input


week_dense_cols = ['week_sell_price', 'diff_week_price', 'week_sell_price_rel_diff', 'week_sell_price_roll_sd7', 'week_sell_price_cumrel',
                   'week_lag_t28', 'week_rolling_mean_t7', 'week_rolling_mean_t30', 'week_rolling_mean_t60', 'week_rolling_mean_t90', 'week_rolling_mean_t180',
                   'week_rolling_std_t7', 'week_rolling_std_t30']
cat_cols = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id']

def main():
  parser = ArgumentParser()
  # train/test hyper parameters
  parser.add_argument('--datapath', type=str, default='data')
  parser.add_argument('--corr_threshold', type=float, default=-0.2)
  parser.add_argument('--test_threshold', type=float, default=-0.1)
  parser.add_argument('--target_scaler', type=float, default=100.0)
  parser.add_argument('--device', type=str, default='cuda:0')
  parser.add_argument('--model_type', type=str, default='dataonly')
  parser.add_argument('--hidden_dim_encoder', type=int, default=64)
  parser.add_argument('--output_dim_encoder', type=int, default=16)
  parser.add_argument('--hidden_dim_db', type=int, default=64)
  parser.add_argument('--epochs', type=int, default=1000, help='default: 200')
  parser.add_argument('--early_stopping_thld', type=int, default=10, help='default: 0 (disabled)')
  parser.add_argument('--valid_freq', type=int, default=1, help='default: 1')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--dual_lr', type=float, default=1.0)
  parser.add_argument('--file', type=str, default='')

  args = parser.parse_args()
  print(args)
  print()

  device = args.device
  seed = args.seed

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  datapath = args.datapath
  corr_threshold = args.corr_threshold
  test_threshold = args.test_threshold
  weekly_filename = 'week_diff_price_demand_corr{}.csv'.format(test_threshold)

  df_input = pd.read_csv(os.path.join(datapath, weekly_filename), index_col=0)
  df_input.head()
  target_scaler = args.target_scaler

  # Rest is used for training
  flag = df_input['wm_yr_wk'] < 11605
  train_x = utils.make_X(df_input[flag], week_dense_cols, cat_cols)
  train_y = df_input['week_sum_demand'][flag]/target_scaler

  # One month of validation data
  flag = (df_input['wm_yr_wk'] < 11609) & (df_input['wm_yr_wk'] >= 11605)
  valid_x = utils.make_X(df_input[flag], week_dense_cols, cat_cols)
  valid_y = df_input['week_sum_demand'][flag]/target_scaler

  # Submission data
  flag = df_input['wm_yr_wk'] >= 11609
  test_x = utils.make_X(df_input[flag], week_dense_cols, cat_cols)
  test_y = df_input['week_sum_demand'][flag]/target_scaler


  # X_train.keys()
  # # Resetting ordinal encoder
  data_info = {'item_id': df_input['item_id'].unique().shape[0],
               'dept_id': df_input['dept_id'].unique().shape[0],
               'store_id': df_input['store_id'].unique().shape[0],
               'cat_id': df_input['cat_id'].unique().shape[0],
               'state_id': df_input['state_id'].unique().shape[0],
               'dense1': len(week_dense_cols)}

  # Tensorize
  def tensorize(dict_input, device=torch.device("cpu")):
    for key, item in dict_input.items():
      if key == 'dense1':
        dict_input[key] = torch.tensor(item, dtype=torch.float32, device=device)
      else:
        dict_input[key] = torch.tensor(item, dtype=torch.int64, device=device)

  tensorize(train_x, device)
  tensorize(valid_x, device)
  tensorize(test_x, device)

  train_y = torch.tensor(train_y.to_numpy(), dtype=torch.float32, device=device)
  valid_y = torch.tensor(valid_y.to_numpy(), dtype=torch.float32, device=device)
  test_y = torch.tensor(test_y.to_numpy(), dtype=torch.float32, device=device)

  name_to_ind = {'dense1': 0, 'item_id': 1, 'dept_id': 2, 'store_id': 3, 'cat_id': 4, 'state_id': 5}
  train_loader = DataLoader(TensorDataset(train_x['dense1'],
                                          train_x['item_id'].squeeze(),
                                          train_x['dept_id'].squeeze(),
                                          train_x['store_id'].squeeze(),
                                          train_x['cat_id'].squeeze(),
                                          train_x['state_id'].squeeze(),
                                          train_y),
                            batch_size=128, shuffle=False)
  valid_x_list = [valid_x['dense1'], valid_x['item_id'].squeeze(), valid_x['dept_id'].squeeze(),
                  valid_x['store_id'].squeeze(), valid_x['cat_id'].squeeze(), valid_x['state_id'].squeeze()]
  test_x_list = [test_x['dense1'], test_x['item_id'].squeeze(), test_x['dept_id'].squeeze(),
                 test_x['store_id'].squeeze(), test_x['cat_id'].squeeze(), test_x['state_id'].squeeze()]

  print("data size: {}/{}/{}".format(len(train_y), len(valid_y), len(test_y)))


  lr = 0.001
  rule_coeff = 0.0
  pert_coeff = 0.1
  scale = 1.0
  reverse = False
  dual_lr = args.dual_lr

  print('model_type: {}, lr: {}, dual_lr: {}, seed: {}'
        .format('ldcdl', lr, dual_lr, seed))

  merge = 'cat'
  input_dim = data_info['dense1']    # Currently, only dense features are used.
  hidden_dim_encoder = args.hidden_dim_encoder
  output_dim_encoder = args.output_dim_encoder
  hidden_dim_db = args.hidden_dim_db

  rule_encoder = RuleEncoder(input_dim, output_dim_encoder, hidden_dim_encoder)
  data_encoder = DataEncoder(input_dim, output_dim_encoder, hidden_dim_encoder)
  model = M5Net(data_info, rule_encoder, data_encoder, name_to_ind, hidden_dim_db).to(device)    # Not residual connection
  dual = InequalityDual(1).to(device)

  pert_feature_ind = 0    # Index of the feature we impose perturbation. Index of price is 0.
  loss_dual_func = lambda x: torch.mean(x)
  loss_task_func = nn.MSELoss()    # return scalar (reduction=mean)
  l1_func = nn.L1Loss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  dual_optimizer = optim.SGD(dual.parameters(), lr=dual_lr)

  epochs = args.epochs
  early_stopping_thld = args.early_stopping_thld
  counter_early_stopping = 1
  valid_freq = args.valid_freq
  log_interval = 500

  saved_filename = 'm5_ldcdl_weekly_demand_pred_with_pert_price-thld{}seed{}dual{}th{}nepoch{}.demo.pt'.format(corr_threshold, seed, dual_lr,early_stopping_thld, epochs)
  saved_filename =  os.path.join('saved_models', saved_filename)
  print('saved_filename: {}\n'.format(saved_filename))
  best_val_loss = float('inf')
  best_val_ratio = 0.0
  best_train_loss = 0.0
  best_train_ratio = 0.0

  # Training
  for epoch in range(1, 1): # no training
    model.train()
    dual.train()
    train_loss_mae = 0
    train_loss_dual = 0
    train_ratio = 0
    total_train_sample = 0
    for batch_idx, batch_data in enumerate(train_loader):
      batch_train_x = batch_data[:-1]    # input features
      batch_train_y = batch_data[-1]

      optimizer.zero_grad()

      alpha = 0.0

      # stable output
      output = model(batch_train_x, alpha=alpha, merge=merge)
      loss_task = loss_task_func(output, batch_train_y)
      loss_mae = l1_func(output, batch_train_y).item()

      # perturbed input and its output
      pert_batch_train_x = []
      for _train_x in batch_train_x:
        pert_batch_train_x.append(_train_x.detach().clone())

      # Perturbations on price. The index of price is 0.
      pert_batch_train_x[0][:, pert_feature_ind] = get_perturbed_input(pert_batch_train_x[0][:, pert_feature_ind], pert_coeff)
      pert_output = model(pert_batch_train_x, alpha=alpha, merge=merge)    # \hat{y}_{p}    predicted sales from perturbed input

      loss_dual = loss_dual_func(dual(pert_output-output))    # pert_output should be less than output
      loss = loss_task + loss_dual
      loss.backward()
      optimizer.step()

      dual_optimizer.zero_grad()
      dual_loss = -loss_dual_func(dual(pert_output.detach() - output.detach()))
      dual_loss.backward()
      dual_optimizer.step()

      ratio = verification(output, pert_output, threshold=0.0).item()

      train_loss_mae = (loss_mae * batch_train_y.shape[0] / (total_train_sample + batch_train_y.shape[0]) +
                        train_loss_mae * total_train_sample / (total_train_sample + batch_train_y.shape[0]))
      train_loss_dual = (loss_dual * batch_train_y.shape[0] / (total_train_sample + batch_train_y.shape[0]) +
                         train_loss_dual * total_train_sample / (total_train_sample + batch_train_y.shape[0]))
      train_ratio = (ratio * batch_train_y.shape[0] / (total_train_sample + batch_train_y.shape[0]) +
                     train_ratio * total_train_sample / (total_train_sample + batch_train_y.shape[0]))
      total_train_sample += batch_train_y.shape[0]


    # Evaluate on validation set
    if epoch % valid_freq == 0:
      model.eval()
      dual.eval()
      alpha = 0.0

      with torch.no_grad():
        target = valid_y
        output = model(valid_x_list, alpha=alpha, merge=merge)
        val_loss_task = loss_task_func(output, target).item()
        val_loss_mae = l1_func(output, target).item()

        # perturbed input and its output
        pert_valid_x = []
        for _valid_x in valid_x_list:
          pert_valid_x.append(_valid_x.detach().clone())
        pert_valid_x[0][:, pert_feature_ind] = get_perturbed_input(pert_valid_x[0][:, pert_feature_ind], pert_coeff)
        pert_output = model(pert_valid_x, alpha=alpha, merge=merge)    # \hat{y}_{p}    predicted sales from perturbed input

        val_loss_dual = loss_dual_func(dual(pert_output-output))
        val_ratio = verification(output, pert_output, threshold=0.0).item()

        val_loss = val_loss_task + val_loss_dual

      if early_stopping_thld > 0:
        if val_loss_task < best_val_loss:
          counter_early_stopping = 1
          best_val_loss = val_loss_task
          best_val_ratio = val_ratio
          best_val_mae = val_loss_mae
          best_train_mae = train_loss_mae
          best_train_ratio = train_ratio
          print('[Valid] Epoch: {} Loss: {:.6f} (alpha: {:.2f})\t Loss(Task): {:.6f}\t Loss(Dual): {:.6f}\t Ratio: {:.4f} best model is updated %%%%'
                .format(epoch, val_loss, alpha, val_loss_task, val_loss_dual, val_ratio))
          torch.save({
            'epoch': epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'dual_state_dict': dual.state_dict(),
            'dual_optimizer_state_dict': dual_optimizer.state_dict()
          }, saved_filename)
        else:
          print('[Valid] Epoch: {} Loss: {:.6f} (alpha: {:.2f})\t Loss(Task): {:.6f}\t Loss(Dual): {:.6f}\t Ratio: {:.4f}({}/{})'
                .format(epoch, val_loss, alpha, val_loss_task, val_loss_dual, val_ratio, counter_early_stopping, early_stopping_thld))
          if counter_early_stopping >= early_stopping_thld:
            break
          else:
            counter_early_stopping += 1
      else:
        best_val_loss = val_loss_task
        best_val_ratio = val_ratio
        best_val_mae = val_loss_mae
        best_train_mae = train_loss_mae
        best_train_ratio = train_ratio
        print('[Valid] Epoch: {} Loss: {:.6f} (alpha: {:.2f})\t Loss(Task): {:.6f}\t Loss(Dual): {:.6f}\t Ratio: {:.4f} best model is updated %%%%'
              .format(epoch, val_loss, alpha, val_loss_task, val_loss_dual, val_ratio))
        torch.save({
          'epoch': epoch,
          'model_state_dict':model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'dual_state_dict': dual.state_dict(),
          'dual_optimizer_state_dict': dual_optimizer.state_dict()
        }, saved_filename)

  # Test
  rule_encoder = RuleEncoder(input_dim, output_dim_encoder, hidden_dim_encoder)
  data_encoder = DataEncoder(input_dim, output_dim_encoder, hidden_dim_encoder)
  model_eval = M5Net(data_info, rule_encoder, data_encoder, name_to_ind, hidden_dim_db).to(device)    # Not residual connection
  dual_eval = InequalityDual(1).to(device)

  checkpoint = torch.load(saved_filename)
  model_eval.load_state_dict(checkpoint['model_state_dict'])
  dual_eval.load_state_dict(checkpoint['dual_state_dict'])

  model_eval.eval()
  with torch.no_grad():
      target = test_y
      output = model_eval(test_x_list, alpha=0.0, merge=merge)
      test_loss_task = loss_task_func(output, target).item()
      test_loss_mae = l1_func(output, target).item()  # sum up batch loss

  print('\n[Test] Average loss: {:.8f} (MAE) \t {:.8f} (MSE)\n'.format(test_loss_mae, test_loss_task))

  alphas = [0.0]
  # perturbed input and its output
  pert_test_x = []
  for _test_x in test_x_list:
    pert_test_x.append(_test_x.detach().clone())
  pert_test_x[0][:, pert_feature_ind] = get_perturbed_input(pert_test_x[0][:, pert_feature_ind], pert_coeff)
  for alpha in alphas:
    model_eval.eval()
    with torch.no_grad():
      target = test_y
      output = model_eval(test_x_list, alpha=0.0, merge=merge)
      test_loss_mae = l1_func(output, target).item()  # sum up batch loss
      pert_output = model_eval(pert_test_x, alpha=0.0, merge=merge)    # \hat{y}_{p}    predicted sales from perturbed input
      test_ratio = verification(output, pert_output, threshold=0.0).item()

    print('[Test] Average loss: {:.8f} (alpha:{})'.format(test_loss_mae, alpha))
    print("[Test] Ratio of verified predictions: {:.6f} (alpha:{})".format(test_ratio, alpha))

  if args.file != '':
    ofd = open(args.file, 'a+')
    ofd.write('{},{},{},{},{},{},{},{}\n'.format(seed, dual_lr, best_train_mae, best_train_ratio, best_val_mae, best_val_ratio, test_loss_mae, test_ratio))
    ofd.close()

if __name__ == '__main__':
  main()
