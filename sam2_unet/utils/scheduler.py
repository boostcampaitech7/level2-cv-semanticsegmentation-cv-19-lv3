from torch.optim import lr_scheduler

class SchedulerSelector():
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def get_scheduler(self, scheduler, **kargs):
        if scheduler == "steplr":
            return lr_scheduler.StepLR(self.optimizer, **kargs)
        elif scheduler == "multisteplr":
            return lr_scheduler.MultiStepLR(self.optimizer, **kargs)
        elif scheduler == "reducelronplateau":
            return lr_scheduler.ReduceLROnPlateau(self.optimizer, **kargs)
        elif scheduler == "cosineannealinglr":
            return lr_scheduler.CosineAnnealingLR(self.optimizer, **kargs)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")