from torch.optim import lr_scheduler

class SchedulerSelector:
    def __init__(self, sched, optimizer, epoch):
        self.optimizer = optimizer
        self.epoch = epoch
        if sched == 'step':
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step=5, gamma=0.6)
        elif sched == 'cosine':
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epoch, eta_min=1e-6)
            
    def get_sched(self):
        return self.scheduler