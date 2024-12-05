import wandb

class WandbLogger:
    def __init__(self, project_name="Hand Bone Image Segmentation", name=None):
        """
        Args:
            project_name (str): wandb project name (default: "Hand Bone Image Segmentation")
            name (str): wandb run name (optional)
        """
        self.project_name = project_name
        self.name = name
        self.run = None

    def initialize(self, config):
        """Initialize wandb run with config"""
        self.run = wandb.init(
            project=self.project_name,
            name=self.name,
            config=config
        )

    def log_metric(self, metrics, step=None):
        """
        Log metrics to wandb
        
        Args:
            metrics (dict): metrics to log
            step (int, optional): step number
        """
        if self.run is not None:
            wandb.log(metrics, step=step)

    def log_batch_metrics(self, loss, extra_info, learning_rate, step=None):
        """
        Log batch-level metrics
        
        Args:
            loss (float): total loss value
            extra_info (dict): dictionary containing detailed losses
            learning_rate (float): current learning rate
            step (int, optional): step number
        """
        metrics = {
            "loss": loss,
            "learning_rate": learning_rate,
            **extra_info
        }
        self.log_metric(metrics, step)

    def log_epoch_metrics(self, metrics):
        """
        Log epoch-level metrics
        
        Args:
            metrics (dict): Dictionary containing all metrics to log
                Expected keys: epoch, loss, cls_loss, angle_loss, iou_loss,
                            learning_rate, epoch_time
        """
        if self.run is not None:
            self.log_metric(metrics)

    def log_model(self, model_path, model_name):
        """
        Log model checkpoint
        
        Args:
            model_path (str): path to model checkpoint
            model_name (str): name of model artifact
        """
        if self.run is not None:
            artifact = wandb.Artifact(
                model_name, 
                type='model',
                description=f"Model checkpoint at {model_path}"
            )
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)

    def finish(self):
        """End the wandb run"""
        if self.run is not None:
            wandb.finish()