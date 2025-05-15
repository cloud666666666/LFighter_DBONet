import torch
from torch.utils import data
import numpy as np


class CustomDataset(data.Dataset):
    def __init__(self, dataset, indices, source_class=None, target_class=None):
        self.dataset = dataset
        self.indices = indices
        self.source_class = source_class
        self.target_class = target_class
        self.contains_source_class = False

    def __getitem__(self, index):
        x, y = self.dataset[int(self.indices[index])]

        # ✅ 修复1：将 y 转换为 1D long Tensor，保证形状一致
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long().view(-1)
        elif isinstance(y, torch.Tensor):
            y = y.view(-1).long()
        else:
            y = torch.tensor([y], dtype=torch.long)

        # ✅ 修复2：处理标签翻转（如定义了 source 和 target）
        if self.source_class is not None and self.target_class is not None:
            if y.item() == self.source_class:
                y = torch.tensor([self.target_class], dtype=torch.long)

        return x, y

    def __len__(self):
        return len(self.indices)


class PoisonedDataset(data.Dataset):
    def __init__(self, dataset, source_class=None, target_class=None):
        self.dataset = dataset
        self.source_class = source_class
        self.target_class = target_class

    def __getitem__(self, index):
        x, y = self.dataset[index]

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long().view(-1)
        elif isinstance(y, torch.Tensor):
            y = y.view(-1).long()
        else:
            y = torch.tensor([y], dtype=torch.long)

        if self.source_class is not None and self.target_class is not None:
            if y.item() == self.source_class:
                y = torch.tensor([self.target_class], dtype=torch.long)

        return x, y

    def __len__(self):
        return len(self.dataset)

    
class IMDBDataset:
    def __init__(self, reviews, targets):
        """
        Argument:
        reviews: a numpy array
        targets: a vector array
        
        Return xtrain and ylabel in torch tensor datatype
        """
        self.reviews = reviews
        self.target = targets
    
    def __len__(self):
        # return length of dataset
        return len(self.reviews)
    
    def __getitem__(self, index):
        # given an index (item), return review and target of that index in torch tensor
        x = torch.tensor(self.reviews[index,:], dtype = torch.long)
        y = torch.tensor(self.target[index], dtype = torch.float)
        
        return  x, y

# A method for combining datasets  
def combine_datasets(list_of_datasets):
    return data.ConcatDataset(list_of_datasets)
    