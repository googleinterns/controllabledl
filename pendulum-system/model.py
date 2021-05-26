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
import torch
import torch.nn as nn
import torch.nn.functional as F


class InequalityDual(nn.Module):
  def __init__(self, in_features):
    super(InequalityDual, self).__init__()
    self.in_features = in_features
    self.weight = nn.Parameter(torch.randn(in_features).abs_())

  def forward(self, x):
    # Inequality dual variables are always positive
    self.weight.data.clamp_(min=0.0)
    return x * self.weight

class EqualityDual(nn.Module):
  def __init__(self, in_features):
    super(EqualityDual, self).__init__()
    self.in_features = in_features
    self.weight = nn.Parameter(torch.randn(in_features))

  def forward(self, x):
    return x * self.weight


class NaiveModel(nn.Module):
  def __init__(self):
    super(NaiveModel, self).__init__()
    self.net = nn.Identity()

  def forward(self, x, alpha=0.0):
    return self.net(x)


class RuleEncoder(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim=4):
    super(RuleEncoder, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                             nn.ReLU(),
                             nn.Linear(hidden_dim, hidden_dim),
                             nn.ReLU(),
                             nn.Linear(hidden_dim, output_dim)
                            )

  def forward(self, x):
    return self.net(x)


class DataEncoder(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim=4):
    super(DataEncoder, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                             nn.ReLU(),
                             nn.Linear(hidden_dim, hidden_dim),
                             nn.ReLU(),
                             nn.Linear(hidden_dim, output_dim)
                            )

  def forward(self, x):
    return self.net(x)


class DataonlyNet(nn.Module):
  def __init__(self, input_dim, output_dim, data_encoder, hidden_dim=4, n_layers=2, skip=False, input_type='state'):
    super(DataonlyNet, self).__init__()
    self.skip = skip
    self.input_type = input_type
    self.data_encoder = data_encoder
    self.n_layers = n_layers
    self.input_dim_decision_block = self.data_encoder.output_dim

    self.net = []
    for i in range(n_layers):
      if i == 0:
        in_dim = self.input_dim_decision_block
      else:
        in_dim = hidden_dim

      if i == n_layers-1:
        out_dim = output_dim
      else:
        out_dim = hidden_dim

      self.net.append(nn.Linear(in_dim, out_dim))
      if i != n_layers-1:
        self.net.append(nn.ReLU())

    self.net = nn.Sequential(*self.net)

  def get_z(self, x, alpha=0.0):
    data_z = self.data_encoder(x)

    return data_z

  def forward(self, x, alpha=0.0):
    # merge: cat or add
    data_z = self.data_encoder(x)
    z = data_z

    if self.skip:
      if self.input_type == 'seq':
        return self.net(z) + x[:,-1,:]
      else:
        return self.net(z) + x    # predict delta values
    else:
      return self.net(z)    # predict absolute values


class Net(nn.Module):
  def __init__(self, input_dim, output_dim, rule_encoder, data_encoder, hidden_dim=4, n_layers=2, merge='cat', skip=False, input_type='state'):
    super(Net, self).__init__()
    self.skip = skip
    self.input_type = input_type
    self.rule_encoder = rule_encoder
    self.data_encoder = data_encoder
    self.n_layers = n_layers
    assert self.rule_encoder.input_dim ==  self.data_encoder.input_dim
    assert self.rule_encoder.output_dim ==  self.data_encoder.output_dim
    self.merge = merge
    if merge == 'cat':
      self.input_dim_decision_block = self.rule_encoder.output_dim * 2
    elif merge == 'add':
      self.input_dim_decision_block = self.rule_encoder.output_dim

    self.net = []
    for i in range(n_layers):
      if i == 0:
        in_dim = self.input_dim_decision_block
      else:
        in_dim = hidden_dim

      if i == n_layers-1:
        out_dim = output_dim
      else:
        out_dim = hidden_dim

      self.net.append(nn.Linear(in_dim, out_dim))
      if i != n_layers-1:
        self.net.append(nn.ReLU())

    self.net = nn.Sequential(*self.net)

  def get_z(self, x, alpha=0.0):
    rule_z = self.rule_encoder(x)
    data_z = self.data_encoder(x)

    if self.merge=='add':
      z = alpha*rule_z + (1-alpha)*data_z    # merge: Add
    elif self.merge=='cat':
      z = torch.cat((alpha*rule_z, (1-alpha)*data_z), dim=-1)    # merge: Concat
    elif self.merge=='equal_cat':
      z = torch.cat((rule_z, data_z), dim=-1)    # merge: Concat

    return z

  def forward(self, x, alpha=0.0):
    # merge: cat or add
    rule_z = self.rule_encoder(x)
    data_z = self.data_encoder(x)

    if self.merge=='add':
      z = alpha*rule_z + (1-alpha)*data_z    # merge: Add
    elif self.merge=='cat':
      z = torch.cat((alpha*rule_z, (1-alpha)*data_z), dim=-1)    # merge: Concat
    elif self.merge=='equal_cat':
      z = torch.cat((rule_z, data_z), dim=-1)    # merge: Concat

    if self.skip:
      if self.input_type == 'seq':
        return self.net(z) + x[:,-1,:]
      else:
        return self.net(z) + x    # predict delta values
    else:
      return self.net(z)    # predict absolute values


class SharedNet(nn.Module):
  def __init__(self, input_dim, output_dim, rule_encoder, data_encoder, hidden_dim=4, n_layers=2, merge='cat', skip=False, input_type='state'):
    super(SharedNet, self).__init__()
    self.skip = skip
    self.input_type = input_type
    self.rule_encoder = rule_encoder
    self.data_encoder = data_encoder
    self.n_layers = n_layers
    assert self.rule_encoder.input_dim ==  self.data_encoder.input_dim
    assert self.rule_encoder.output_dim ==  self.data_encoder.output_dim
    self.merge = merge
    if merge == 'cat':
      self.input_dim_decision_block = self.rule_encoder.output_dim * 2
    elif merge == 'add':
      self.input_dim_decision_block = self.rule_encoder.output_dim
    self.shared_net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, self.rule_encoder.input_dim))
    self.net = []
    for i in range(n_layers):
      if i == 0:
        in_dim = self.input_dim_decision_block
      else:
        in_dim = hidden_dim

      if i == n_layers-1:
        out_dim = output_dim
      else:
        out_dim = hidden_dim

      self.net.append(nn.Linear(in_dim, out_dim))
      if i != n_layers-1:
        self.net.append(nn.ReLU())

    self.net = nn.Sequential(*self.net)

  def get_z(self, x, alpha=0.0):
    out = self.shared_net(x)

    rule_z = self.rule_encoder(out)
    data_z = self.data_encoder(out)

    if self.merge=='add':
      z = alpha*rule_z + (1-alpha)*data_z    # merge: Add
    elif self.merge=='cat':
      z = torch.cat((alpha*rule_z, (1-alpha)*data_z), dim=-1)    # merge: Concat
    elif self.merge=='equal_cat':
      z = torch.cat((rule_z, data_z), dim=-1)    # merge: Concat

    return z

  def forward(self, x, alpha=0.0):
    # merge: cat or add
    out = self.shared_net(x)

    rule_z = self.rule_encoder(out)
    data_z = self.data_encoder(out)

    if self.merge=='add':
      z = alpha*rule_z + (1-alpha)*data_z    # merge: Add
    elif self.merge=='cat':
      z = torch.cat((alpha*rule_z, (1-alpha)*data_z), dim=-1)    # merge: Concat
    elif self.merge=='equal_cat':
      z = torch.cat((rule_z, data_z), dim=-1)    # merge: Concat

    if self.skip:
      if self.input_type == 'seq':
        return self.net(z) + x[:,-1,:]
      else:
        return self.net(z) + x    # predict delta values
    else:
      return self.net(z)    # predict absolute values
