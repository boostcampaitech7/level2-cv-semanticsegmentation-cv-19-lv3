import torch.optim as optim

class OptimizerSelector():
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.lr = lr

    def get_optimizer(self, optimizer, **kargs):
        if optimizer == "adam":
            return optim.Adam(self.model.parameters(), self.lr, **kargs)
        elif optimizer == "adamw":
            return optim.AdamW(self.model.parameters(), self.lr, **kargs)
        elif optimizer == "sgd":
            return optim.SGD(self.model.parameters(), self.lr, **kargs)
        elif optimizer == "rmsprop":
            return optim.RMSprop(self.model.parameters(), self.lr, **kargs)
        elif optimizer == "nadam":
            return optim.NAdam(self.model.parameters(), self.lr, **kargs)
        elif optimizer == "radam":
            return optim.RAdam(self.model.parameters(), self.lr, **kargs)
        elif optimizer == "adagrad":
            return optim.Adagrad(self.model.parameters(), self.lr, **kargs)
        elif optimizer == "adadelta":
            return optim.Adadelta(self.model.parameters(), self.lr, **kargs)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")