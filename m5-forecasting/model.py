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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RuleEncoder(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim=4):
    super(RuleEncoder, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim
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
    self.hidden_dim = hidden_dim
    self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                             nn.ReLU(),
                             nn.Linear(hidden_dim, hidden_dim),
                             nn.ReLU(),
                             nn.Linear(hidden_dim, output_dim)
                            )

  def forward(self, x):
    return self.net(x)

class M5Net(nn.Module):
  def __init__(self, data_info, rule_encoder, data_encoder, name_to_ind, hidden_dim=16):
    super(M5Net, self).__init__()
    self.data_info = data_info
    self.name_to_ind = name_to_ind
    self.emb_dict = nn.ModuleDict()

    self.rule_encoder = rule_encoder
    self.data_encoder = data_encoder
    self.input_dim = self.rule_encoder.output_dim + self.data_encoder.output_dim
    self.net = nn.Sequential(nn.Linear(self.input_dim, hidden_dim),
                             nn.ReLU(),
                             nn.Linear(hidden_dim, 1))
        
  def get_z(self, x, alpha=0.0):
    pass
        
  def forward(self, x, alpha=0.0, merge='cat'):
    # merge: cat or add
    input_dense_cat = [x[self.name_to_ind['dense1']]]    # index 0 is 'dense1'
    x = torch.cat(input_dense_cat, dim=-1)

    rule_z = self.rule_encoder(x)
    data_z = self.data_encoder(x)
    
    if merge=='add':
      z = alpha*rule_z + (1-alpha)*data_z    # merge: Add
    elif merge=='cat':
      z = torch.cat((alpha*rule_z, (1-alpha)*data_z), dim=-1)    # merge: Concat
    
    return self.net(z).squeeze()    # predict absolute values

class Net(nn.Module):
  def __init__(self, data_info, emb_dim=1, activation=nn.Tanh(), num_cat_features=12):
    super(Net, self).__init__()
    self.data_info = data_info
    self.emb_dim = emb_dim
    self.activation = activation
    self.dense_dim = data_info['dense1']
    self.input_dim = self.dense_dim + emb_dim*(num_cat_features-1) + 3*emb_dim
        
    self.net = nn.Sequential(nn.Linear(self.input_dim, 150),
                             self.activation,
                             nn.Linear(150, 75),
                             self.activation,
                             nn.Linear(75, 10),
                             self.activation,
                             nn.Linear(10, 1)
                            )

    temp_moduledict = {}
    self.cat_keys = []
    for key, val in self.data_info.items():
      if key == 'dense1':
        continue
      elif key == 'item_id':
        temp_moduledict[key] = nn.Embedding(val, 3*self.emb_dim)
      else:
        temp_moduledict[key] = nn.Embedding(val, self.emb_dim)
      self.cat_keys.append(key)

    self.emb_layers = nn.ModuleDict(temp_moduledict)

  def forward(self, dict_data):
    emb_out = [dict_data['dense1']]
    for key in self.cat_keys:
      emb_out.append(self.emb_layers[key](dict_data[key]).squeeze(1))    # emb input shape: (batch_size, 1)

    x = torch.cat(emb_out, dim=1)
    return self.net(x)
    
    
    