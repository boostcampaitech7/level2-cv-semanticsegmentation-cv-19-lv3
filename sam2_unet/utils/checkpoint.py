import torch

def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath) 

def load_checkpoint(filepath, model, optimizer, scheduler):
    checkpoint = torch.load(filepath, map_location='cpu')
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dice(checkpoint['scheduler_state_dict'])
    loss = checkpoint['loss']

    return epoch, model, optimizer, scheduler, loss