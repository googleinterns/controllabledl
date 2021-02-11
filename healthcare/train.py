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

from sklearn.preprocessing import OneHotEncoder, Normalizer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.beta import Beta

from model import RuleEncoder, DataEncoder, Net
from utils_learning import verification, get_perturbed_input
from utils_cardio import *


model_info = {'dataonly': {'rule': 0.0},
              'ours-beta1.0': {'beta': [1.0], 'scale': 1.0, 'lr': 0.001},
              'ours-beta0.1': {'beta': [0.1], 'scale': 1.0, 'lr': 0.001},
              'ours-beta0.1-scale0.1': {'beta': [0.1], 'scale': 0.1},
              'ours-beta0.1-scale0.01': {'beta': [0.1], 'scale': 0.01},
              'ours-beta0.1-scale0.05': {'beta': [0.1], 'scale': 0.05},
              'ours-beta0.1-pert0.001': {'beta': [0.1], 'pert': 0.001},
              'ours-beta0.1-pert0.01': {'beta': [0.1], 'pert': 0.01},
              'ours-beta0.1-pert0.1': {'beta': [0.1], 'pert': 0.1},
              'ours-beta0.1-pert1.0': {'beta': [0.1], 'pert': 1.0},
             }


def main():
  parser = ArgumentParser()
  parser.add_argument('--datapath', type=str, default='data/cardio_train.csv')
  parser.add_argument('--rule_threshold', type=float, default=129.5)
  parser.add_argument('--src_usual_ratio', type=float, default=0.3)
  parser.add_argument('--src_unusual_ratio', type=float, default=0.7)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--device', type=str, default='cuda:0')
  parser.add_argument('--target_rule_ratio', type=float, default=0.7)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--train_ratio', type=float, default=0.7)
  parser.add_argument('--validation_ratio', type=float, default=0.1)
  parser.add_argument('--test_ratio', type=float, default=0.2)
  parser.add_argument('--model_type', type=str, default='dataonly')
  parser.add_argument('--input_dim_encoder', type=int, default=16)
  parser.add_argument('--output_dim_encoder', type=int, default=16)
  parser.add_argument('--hidden_dim_encoder', type=int, default=100)
  parser.add_argument('--hidden_dim_db', type=int, default=16)
  parser.add_argument('--n_layers', type=int, default=1)
  parser.add_argument('--rule_ind', type=int, default=5)  
  parser.add_argument('--epochs', type=int, default=1000, help='default: 1000')
  parser.add_argument('--early_stopping_thld', type=int, default=10, help='default: 10')
  parser.add_argument('--valid_freq', type=int, default=1, help='default: 1')

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

  # Load dataset
  df = pd.read_csv(os.path.join(datapath), delimiter=';')
  df = df.drop(['id'], axis=1)

  y = df['cardio']
  X_raw = df.drop(['cardio'], axis=1)

  print("Target class ratio:")
  print("# of cardio=1: {}/{} ({:.2f}%)".format(np.sum(y==1), len(y), 100*np.sum(y==1)/len(y)))
  print("# of cardio=0: {}/{} ({:.2f}%)\n".format(np.sum(y==0), len(y), 100*np.sum(y==0)/len(y)))

  column_trans = ColumnTransformer(
      [('age_norm', StandardScaler(), ['age']),
       ('height_norm', StandardScaler(), ['height']),
       ('weight_norm', StandardScaler(), ['weight']),
       ('gender_cat', OneHotEncoder(), ['gender']),
       ('ap_hi_norm', StandardScaler(), ['ap_hi']),
       ('ap_lo_norm', StandardScaler(), ['ap_lo']),
       ('cholesterol_cat', OneHotEncoder(), ['cholesterol']),
       ('gluc_cat', OneHotEncoder(), ['gluc']),
       ('smoke_cat', OneHotEncoder(), ['smoke']),
       ('alco_cat', OneHotEncoder(), ['alco']),
       ('active_cat', OneHotEncoder(), ['active']),
      ], remainder='passthrough'
  )

  X = column_trans.fit_transform(X_raw)
  num_samples = X.shape[0]
  X_np = X.copy()

  # Rule : higher ap -> higher risk
  rule_threshold = args.rule_threshold
  rule_ind = args.rule_ind
  rule_feature = 'ap_hi'

  low_ap_negative = (df[rule_feature] <= rule_threshold) & (df['cardio'] == 0)    # usual
  high_ap_positive = (df[rule_feature] > rule_threshold) & (df['cardio'] == 1)    # usual
  low_ap_positive = (df[rule_feature] <= rule_threshold) & (df['cardio'] == 1)    # unusual
  high_ap_negative = (df[rule_feature] > rule_threshold) & (df['cardio'] == 0)    # unusual

  # Samples in Usual group
  X_usual = X[low_ap_negative | high_ap_positive]
  y_usual = y[low_ap_negative | high_ap_positive]
  y_usual = y_usual.to_numpy()
  X_usual, y_usual = shuffle(X_usual, y_usual, random_state=0)
  num_usual_samples = X_usual.shape[0]

  # Samples in Unusual group
  X_unusual = X[low_ap_positive | high_ap_negative]
  y_unusual = y[low_ap_positive | high_ap_negative]
  y_unusual = y_unusual.to_numpy()
  X_unusual, y_unusual = shuffle(X_unusual, y_unusual, random_state=0)
  num_unusual_samples = X_unusual.shape[0]

  # Build a source dataset
  src_usual_ratio = args.src_usual_ratio
  src_unusual_ratio = args.src_unusual_ratio
  num_samples_from_unusual = int(src_unusual_ratio * num_unusual_samples)
  num_samples_from_usual = int(num_samples_from_unusual * src_usual_ratio / (1-src_usual_ratio))

  X_src = np.concatenate((X_usual[:num_samples_from_usual], X_unusual[:num_samples_from_unusual]), axis=0)
  y_src = np.concatenate((y_usual[:num_samples_from_usual], y_unusual[:num_samples_from_unusual]), axis=0)
  print()
  print("Source dataset statistics:")
  print("# of samples in Usual group: {}".format(num_samples_from_usual))
  print("# of samples in Unusual group: {}".format(num_samples_from_unusual))
  print("Usual ratio: {:.2f}%".format(100 * num_samples_from_usual / (X_src.shape[0])))

  train_ratio = args.train_ratio
  validation_ratio = args.validation_ratio
  test_ratio = args.test_ratio
  train_X, test_X, train_y, test_y = train_test_split(X_src, y_src, test_size=1 - train_ratio, random_state=seed)
  valid_X, test_X, valid_y, test_y = train_test_split(test_X, test_y, test_size=test_ratio / (test_ratio + validation_ratio), random_state=seed)

  train_X, train_y = torch.tensor(train_X, dtype=torch.float32, device=device), torch.tensor(train_y, dtype=torch.float32, device=device)
  valid_X, valid_y = torch.tensor(valid_X, dtype=torch.float32, device=device), torch.tensor(valid_y, dtype=torch.float32, device=device)
  test_X, test_y = torch.tensor(test_X, dtype=torch.float32, device=device), torch.tensor(test_y, dtype=torch.float32, device=device)

  batch_size = args.batch_size
  train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
  valid_loader = DataLoader(TensorDataset(valid_X, valid_y), batch_size=valid_X.shape[0])
  test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=test_X.shape[0])
  print("data size: {}/{}/{}".format(len(train_X), len(valid_X), len(test_X)))

  model_type = args.model_type
  if model_type not in model_info:
    # default setting
    lr = 0.001
    pert_coeff = 0.1
    scale = 1.0
    beta_param = [1.0]
    alpha_distribution = Beta(float(beta_param[0]), float(beta_param[0]))
    model_params = {}

  else:
    model_params = model_info[model_type]
    lr = model_params['lr'] if 'lr' in model_params else 0.001
    pert_coeff = model_params['pert'] if 'pert' in model_params else 0.1
    scale = model_params['scale'] if 'scale' in model_params else 1.0
    beta_param = model_params['beta'] if 'beta' in model_params else [1.0]

    if len(beta_param) == 1:
      alpha_distribution = Beta(float(beta_param[0]), float(beta_param[0]))
    elif len(beta_param) == 2:
      alpha_distribution = Beta(float(beta_param[0]), float(beta_param[1]))

  print('model_type: {}\tscale:{}\tBeta distribution: Beta({})\tlr: {}\t \tpert_coeff: {}'.format(model_type, scale, beta_param, lr, pert_coeff))

  merge = 'cat'
  input_dim = 19
  output_dim_encoder = args.output_dim_encoder
  hidden_dim_encoder = args.hidden_dim_encoder
  hidden_dim_db = args.hidden_dim_db
  n_layers = args.n_layers
  output_dim = 1

  rule_encoder = RuleEncoder(input_dim, output_dim_encoder, hidden_dim_encoder)
  data_encoder = DataEncoder(input_dim, output_dim_encoder, hidden_dim_encoder)
  model = Net(input_dim, output_dim, rule_encoder, data_encoder, hidden_dim=hidden_dim_db, n_layers=n_layers, merge=merge).to(device)    # Not residual connection

  optimizer = optim.Adam(model.parameters(), lr=lr)        
  loss_rule_func = lambda x,y: torch.mean(F.relu(x-y))    # if x>y, penalize it.
  loss_task_func = nn.BCELoss()    # return scalar (reduction=mean)

  epochs = args.epochs
  early_stopping_thld = args.early_stopping_thld
  counter_early_stopping = 1
  valid_freq = args.valid_freq  

  saved_filename = 'cardio_{}_rule-{}_src{}-target{}_seed{}.demo.pt'.format(model_type, rule_feature, src_usual_ratio, src_usual_ratio, seed)
  saved_filename =  os.path.join('saved_models', saved_filename)
  print('saved_filename: {}\n'.format(saved_filename))
  best_val_loss = float('inf')

  for epoch in range(1, epochs+1):
    model.train()
    for batch_train_x, batch_train_y in train_loader:
      batch_train_y = batch_train_y.unsqueeze(-1)

      optimizer.zero_grad()

      if model_type.startswith('dataonly'):
        alpha = 0.0
      elif model_type.startswith('ruleonly'):
        alpha = 1.0
      elif model_type.startswith('ours'):
        alpha = alpha_distribution.sample().item()

      # stable output
      output = model(batch_train_x, alpha=alpha)
      loss_task = loss_task_func(output, batch_train_y)

      # perturbed input and its output
      pert_batch_train_x = batch_train_x.detach().clone()
      pert_batch_train_x[:,rule_ind] = get_perturbed_input(pert_batch_train_x[:,rule_ind], pert_coeff)
      pert_output = model(pert_batch_train_x, alpha=alpha)

      loss_rule = loss_rule_func(output, pert_output)    # output should be less than pert_output

      loss = alpha * loss_rule + scale * (1 - alpha) * loss_task

      loss.backward()
      optimizer.step()

    # Evaluate on validation set
    if epoch % valid_freq == 0:
      model.eval()
      if  model_type.startswith('ruleonly'):
        alpha = 1.0
      else:
        alpha = 0.0

      with torch.no_grad():
        for val_x, val_y in valid_loader:
          val_y = val_y.unsqueeze(-1)

          output = model(val_x, alpha=alpha)
          val_loss_task = loss_task_func(output, val_y).item()

          # perturbed input and its output
          pert_val_x = val_x.detach().clone()
          pert_val_x[:,rule_ind] = get_perturbed_input(pert_val_x[:,rule_ind], pert_coeff)
          pert_output = model(pert_val_x, alpha=alpha)    # \hat{y}_{p}    predicted sales from perturbed input

          val_loss_rule = loss_rule_func(output, pert_output).item()
          val_ratio = verification(pert_output, output, threshold=0.0).item()

          val_loss = val_loss_task

          y_true = val_y.cpu().numpy()
          y_score = output.cpu().numpy()
          y_pred = np.round(y_score)
          val_acc = 100 * accuracy_score(y_true, y_pred)

        if val_loss < best_val_loss:
          counter_early_stopping = 1
          best_val_loss = val_loss
          best_model_state_dict = deepcopy(model.state_dict())
          print('[Valid] Epoch: {} Loss: {:.6f} (alpha: {:.2f})\t Loss(Task): {:.6f} Acc: {:.2f}\t Loss(Rule): {:.6f}\t Ratio: {:.4f} best model is updated %%%%'
                .format(epoch, best_val_loss, alpha, val_loss_task, val_acc, val_loss_rule, val_ratio))
          torch.save({
              'epoch': epoch,
              'model_state_dict': best_model_state_dict,
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': best_val_loss
          }, saved_filename)
        else:
          print('[Valid] Epoch: {} Loss: {:.6f} (alpha: {:.2f})\t Loss(Task): {:.6f} Acc: {:.2f}\t Loss(Rule): {:.6f}\t Ratio: {:.4f}({}/{})'
                .format(epoch, val_loss, alpha, val_loss_task, val_acc, val_loss_rule, val_ratio, counter_early_stopping, early_stopping_thld))
          if counter_early_stopping >= early_stopping_thld:
            break
          else:
            counter_early_stopping += 1

  # Test
  rule_encoder = RuleEncoder(input_dim, output_dim_encoder, hidden_dim_encoder)
  data_encoder = DataEncoder(input_dim, output_dim_encoder, hidden_dim_encoder)
  model_eval = Net(input_dim, output_dim, rule_encoder, data_encoder, hidden_dim=hidden_dim_db, n_layers=n_layers, merge=merge).to(device)    # Not residual connection

  checkpoint = torch.load(saved_filename)
  model_eval.load_state_dict(checkpoint['model_state_dict'])
  print("best model loss: {:.6f}\t at epoch: {}".format(checkpoint['loss'], checkpoint['epoch']))

  model_eval.eval()
  with torch.no_grad():
    for te_x, te_y in test_loader:
      te_y = te_y.unsqueeze(-1)

    output = model_eval(te_x, alpha=0.0)
    test_loss_task = loss_task_func(output, te_y).item()
  print('\n[Test] Average loss: {:.8f}\n'.format(test_loss_task))

  model_eval.eval()
  alphas = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
  # perturbed input and its output
  pert_test_x = te_x.detach().clone()
  pert_test_x[:,rule_ind] = get_perturbed_input(pert_test_x[:,rule_ind], pert_coeff)
  for alpha in alphas:
    model_eval.eval()
    with torch.no_grad():
      for te_x, te_y in test_loader:
        te_y = te_y.unsqueeze(-1)

      if model_type.startswith('dataonly'):
        output = model_eval(te_x, alpha=0.0)
      elif model_type.startswith('ours'):
        output = model_eval(te_x, alpha=alpha)
      elif model_type.startswith('ruleonly'):
        output = model_eval(te_x, alpha=1.0)

      test_loss_task = loss_task_func(output, te_y).item()

      if model_type.startswith('dataonly'):
        pert_output = model_eval(pert_test_x, alpha=0.0)
      elif model_type.startswith('ours'):
        pert_output = model_eval(pert_test_x, alpha=alpha)
      elif model_type.startswith('ruleonly'):
        pert_output = model_eval(pert_test_x, alpha=1.0)

      test_ratio = verification(pert_output, output, threshold=0.0).item()

      y_true = te_y.cpu().numpy()
      y_score = output.cpu().numpy()
      y_pred = np.round(y_score)
      test_acc = accuracy_score(y_true, y_pred)

    print('[Test] Average loss: {:.8f} (alpha:{})'.format(test_loss_task, alpha))
    print('[Test] Accuracy: {:.4f} (alpha:{})'.format(test_acc, alpha))
    print("[Test] Ratio of verified predictions: {:.6f} (alpha:{})".format(test_ratio, alpha))
    print()
        
if __name__ == '__main__':
  main()
