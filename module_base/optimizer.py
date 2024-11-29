import torch.optim as optim
from lion_pytorch import Lion
from adamp import AdamP

class OptimizerSelector:
    def __init__(self, opt, model, lr, weight_decay, betas):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        if opt == 'Adam': 
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif opt == 'AdamW': 
            self.optimizer = optim.AdamW(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif opt == 'Lion': 
            self.optimizer = Lion(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif opt == 'AdamP':
            optimizer = AdamP(params=self.model.parameters(), lr=self.lr, betas = self.betas, weight_decay=self.weight_decay)
            
    def get_optim(self):
        return self.optimizer