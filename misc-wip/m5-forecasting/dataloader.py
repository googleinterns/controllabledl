import torch
from torch.utils.data import Dataset


class M5Dataset(Dataset):
    def __init__(self, dict_data):
        self.dict_data = dict_data
        self.keys = dict_data.keys()
        self.len = len(dict_data['dense1'])
        
        print("M5 dataset keys: {}".format(self.keys))
    
    def __getitem__(self, index):
        ret = {}
        for key in self.keys:
            ret[key] = self.dict_data[key][index]
        
        return ret
    
    def __len__(self):
        return self.len
    


def transform_tensor_type(X_train, device):
    ret = {}
    for key, val in X_train.items():
        if key == 'dense1':
            ret[key] = torch.tensor(val, dtype=torch.float, device=device)
        else:
            ret[key] = torch.tensor(val, dtype=torch.long, device=device)
            
    return ret