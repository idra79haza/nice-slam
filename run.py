import argparse
import random

import numpy as np
import torch

from src import config
from src.iLabel_plus_SLAM import iLabel_plus_SLAM


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # setup_seed(20)

    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*/iLabel++.'
    )
    parser.add_argument('--config', type=str, default='configs/Replica/room0_ilabel_plus.yaml', help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--dataset_type', type=str, default="replica", choices= ["replica", "replica_nyu_cnn", "scannet"], 
                        help='the dataset to be used,')
    # parser.add_argument('--run_type', type=str, default="ilabel_plus", choices= ["ilabel_plus", "nice", "imap"], 
    #                     help='type of SLAM')
    ilabel_plus_parser = parser.add_mutually_exclusive_group(required=False)
    # ilabel_plus_parser.add_argument('--nice', dest='nice', action='store_true')
    # ilabel_plus_parser.add_argument('--imap', dest='nice', action='store_false')
    ilabel_plus_parser.add_argument('--ilabel_plus', dest='ilabel_plus', action='store_true')
    

    parser.set_defaults(ilabel_plus=True)
    parser.set_defaults(nice=True)
    args = parser.parse_args()

    if args.ilabel_plus:
        config_to_load = 'configs/ilabel_plus.yaml'
    elif args.nice:
        config_to_load = 'configs/nice_slam.yaml'
    else:
        config_to_load = 'configs/imap.yaml'

    cfg = config.load_config(args.config, config_to_load)

    slam = iLabel_plus_SLAM(cfg, args)

    slam.run()

if __name__ == '__main__':
    main()
