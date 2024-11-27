from mmengine.config import Config
from mmengine.runner import Runner


##############################
# 데이터 경로를 입력하세요

CONFIG_PATH = "/data/ephemeral/home/parkjunil/level2-cv-semanticsegmentation-cv-19-lv3/mmsegmentation/configs/_baseline_/config_for_this_example.py"
CHECKPOINT_PATH = "/data/ephemeral/home/parkjunil/work_dir/hrnet18_upernet_2048_elas_rotate_hflip_whole/best_mDice_iter_18000.pth"
SUBMISSION_PATH = "/data/ephemeral/home/submission"
WORK_DIR = "/data/ephemeral/home/parkjunil/work_dir/hrnet18_upernet_2048_elas_rotate_hflip_whole_inference"
TTA = False


def load_config():
    # load config
    cfg = Config.fromfile(CONFIG_PATH)
    cfg.launcher = "none"
    cfg.work_dir = WORK_DIR

    # resume training
    cfg.resume = False
    return cfg

def train():
    # load config
    cfg =load_config()
    # cfg.load_from = CHECKPOINT_PATH
    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()


def inference():
    cfg = load_config()
    cfg.load_from = CHECKPOINT_PATH
    if TTA:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model
    
    # inference시에는 wandb 비활성화
    cfg.vis_backends = [dict(type='LocalVisBackend')]
    cfg.log_config = dict(
        interval=1,
        hooks=[
            dict(type='TextLoggerHook', by_epoch=False)
        ]
        )
    
    
    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == "__main__":
    train()
    # inference()