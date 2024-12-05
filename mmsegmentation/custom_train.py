from mmengine.config import Config
from mmengine.runner import Runner
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', help='config의 경로', default='./configs/_baseline_/config_for_this_example.py')
    parser.add_argument('--work_dir', help='실험 결과를 저장할 경로', default='./work_dir')
    args = parser.parse_args()
    return args

    
def main(config_path, work_dir):
     # load config
    cfg = Config.fromfile(config_path)
    cfg.launcher = "none"
    cfg.work_dir = work_dir

    # resume training
    cfg.resume = False
    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.config_path, args.work_dir)