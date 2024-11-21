from mmengine.config import Config
from mmengine.runner import Runner
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', help='config의 경로', default='./configs/_baseline_/config_for_this_example.py')
    parser.add_argument('--checkpoint_path', help='.pth파일 위치', default='./work_dir/best_mDice_iter_20000.pth')
    parser.add_argument('--work_dir', help='실험 결과를 저장할 경로', default='./work_dir')
    parser.add_argument('--submission_path', help='inference 결과를 저장할 경로', default='./result')
    args = parser.parse_args()
    return args

    
def main(config_path, checkpoint_path, work_dir, submission_path):
     # load config
    cfg = Config.fromfile(config_path)
    cfg.launcher = "none"
    cfg.work_dir = work_dir

    # resume training
    cfg.resume = False
    cfg.load_from = checkpoint_path
    cfg.test_evaluator.save_path = submission_path
    
    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.config_path, args.checkpoint_path, args.work_dir, args.submission_path)