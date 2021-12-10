import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from utils import experiment_manager, networks, datasets
FONTSIZE = 16

AOI_IDS = [
]


def qualitative_comparison(config_names: list, run_type: str = 'validation'):
    pass


def qualitative_comparison_assembled(config_names: list):
    pass


def assessment_argument_parser():
    # https://docs.python.org/3/library/argparse.html#the-add-argument-method
    parser = argparse.ArgumentParser(description="Experiment Args")

    parser.add_argument('-c', "--config-file", dest='config_file', required=True, help="path to config file")
    parser.add_argument('-o', "--output-dir", dest='output_dir', required=True, help="path to output directory")
    parser.add_argument('-d', "--dataset-dir", dest='dataset_dir', default="", required=True,
                        help="path to output directory")
    parser.add_argument('-r', "--run-type", dest='run_type', default="validation", required=False, help="run type")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    args = assessment_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    qualitative_assessment_change(cfg, run_type=args.run_type)
    qualitative_assessment_sem(cfg, run_type=args.run_type)