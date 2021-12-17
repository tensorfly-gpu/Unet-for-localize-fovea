import paddle
import paddle.nn as nn
from paddle.io import Dataset
from paddle.nn import Linear,Dropout,BatchNorm1D
import paddle.nn.functional as F

import numpy as np
import random
random.seed(1024)

class TrainDataset(Dataset):
    def __init__(self,train_features,labels,sample_list):
        super(TrainDataset,self).__init__()
        self.num_samples = 100
        self.train_features = train_features
        self.labels = labels
        self.sample_list = sample_list

    def __getitem__(self, item):
        item = self.sample_list[item]
        features = np.array(self.train_features[item]).astype('float32')
        labels = np.array(self.labels[item]).astype('float32')
        return features,labels

    def __len__(self):
        return self.num_samples