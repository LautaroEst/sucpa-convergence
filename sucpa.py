

import torch
from torch import nn


class SUCPA(nn.Module):
    
    def __init__(self, num_classes, steps=10):
        super().__init__()
        self.num_classes = num_classes
        self.steps = steps
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