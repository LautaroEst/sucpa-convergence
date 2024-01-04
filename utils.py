

import torch
from torch import nn
import numpy as np

dataset2numclasses = {
    "sst2": 2,
    "mnli": 3,
    "dogs-vs-cats": 2,
}

def load_data(dataset):
    rs = np.random.RandomState(92893)
    logits = np.load(f'data/{dataset}/train_logits.npy')
    labels = np.load(f'data/{dataset}/train_labels.npy')
    idx = rs.choice(logits.shape[0], size=1000, replace=False)
    logits = torch.from_numpy(logits[idx])
    labels = torch.from_numpy(labels[idx])
    return logits, labels


class SUCPA(nn.Module):
    
    def __init__(self, num_classes, steps=10, beta_init=None):
        super().__init__()
        self.num_classes = num_classes
        self.steps = steps
        if beta_init is not None:
            if beta_init.shape != (num_classes,):
                raise ValueError(f"beta_init must be of shape ({num_classes},), but got {beta_init.shape}")
            self.beta = nn.Parameter(beta_init, requires_grad=False)
        else:
            self.beta = nn.Parameter(torch.zeros(num_classes), requires_grad=False)
        self.jacobian = torch.zeros(num_classes, num_classes)
        self.beta_history = None
        self.jacobian_history = None

    def fit(self, logits, class_samples):
        beta_history = [self.beta.data]
        jacobian_history = [self.jacobian]
        for i in range(self.steps):
            log_den = torch.logsumexp(logits + self.beta, dim=1)
            log_sum_all = torch.logsumexp(logits.T - log_den, dim=1)
            self.beta.data = -log_sum_all + torch.log(class_samples)
            beta_history.append(self.beta.data)
            
            probs = torch.softmax(logits, dim=1)
            exp_beta = torch.exp(self.beta)
            den = probs @ exp_beta.unsqueeze(1)
            probs_den = probs / den
            self.jacobian = (probs_den.T @ torch.softmax(logits + self.beta, dim=1)) / probs_den.sum(dim=0).unsqueeze(1)
            jacobian_history.append(self.jacobian)
        self.beta_history = torch.stack(beta_history)
        self.jacobian_history = torch.stack(jacobian_history)
        return self
    
    def calibrate(self, logits):
        return logits + self.beta


