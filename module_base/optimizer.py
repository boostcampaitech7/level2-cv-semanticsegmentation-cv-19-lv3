import torch.optim as optim
from lion_pytorch import Lion

class OptimizerSelector:
    def __init__(self, opt, model, lr, weight_decay):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        if opt == 'Adam': 
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif opt == 'AdamW': 
            self.optimizer = optim.AdamW(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif opt == 'Lion': 
            self.optimizer = Lion(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            
    def get_optim(self):
        return self.optimizer