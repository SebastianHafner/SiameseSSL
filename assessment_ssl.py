import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from utils import experiment_manager, networks, datasets
FONTSIZE = 16


def qualitative_assessment_change(cfg: experiment_manager.CfgNode, run_type: str = 'validation'):
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, 'cpu')
    net.eval()
    ds = datasets.SpaceNet7CDDataset(cfg, run_type, dataset_mode='first_last', no_augmentations=True,
                                     disable_unlabeled=True, disable_multiplier=True)

    for item in ds:
        aoi_id = item['aoi_id']
        x_t1 = item['x_t1']
        x_t2 = item['x_t2']
        logits_change, logits_sem_t1, logits_sem_t2 = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))

        fig, axs = plt.subplots(1, 5, figsize=(20, 4))

        axs[0].imshow(x_t1.numpy().transpose((1, 2, 0)))
        axs[0].set_title(r'Planet $t_1$', fontsize=FONTSIZE)
        axs[1].imshow(x_t2.numpy().transpose((1, 2, 0)))
        axs[1].set_title(r'Planet $t_2$', fontsize=FONTSIZE)

        gt_change = item['y_change'].squeeze()
        axs[2].imshow(gt_change.numpy(), cmap='gray')
        axs[2].set_title(r'GT', fontsize=FONTSIZE)

        logits_change_sem = net.outc_sem_change(torch.cat((logits_sem_t1, logits_sem_t2), dim=1))
        y_pred_change_sem = torch.sigmoid(logits_change_sem).squeeze().detach()
        axs[3].imshow(y_pred_change_sem.numpy(), cmap='gray')
        axs[3].set_title(r'Change Sem', fontsize=FONTSIZE)

        y_pred_change = torch.sigmoid(logits_change).squeeze().detach()
        axs[4].imshow(y_pred_change.numpy(), cmap='gray')
        axs[4].set_title(r'Change', fontsize=FONTSIZE)

        for _, ax in np.ndenumerate(axs):
            ax.set_axis_off()
        plt.tight_layout()

        out_file = Path(cfg.PATHS.OUTPUT) / 'plots' / 'assessment_change_ssl' / f'{aoi_id}.png'
        out_file.parent.mkdir(exist_ok=True)
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)


def qualitative_assessment_sem(cfg: experiment_manager.CfgNode, run_type: str = 'validation'):
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, 'cpu')
    net.eval()
    ds = datasets.SpaceNet7CDDataset(cfg, run_type, dataset_mode='first_last', no_augmentations=True,
                                     disable_unlabeled=True, disable_multiplier=True)

    for item in ds:
        aoi_id = item['aoi_id']
        x_t1 = item['x_t1']
        x_t2 = item['x_t2']
        _, logits_sem_t1, logits_sem_t2 = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))

        fig, axs = plt.subplots(2, 3, figsize=(12, 8))

        axs[0, 0].imshow(x_t1.numpy().transpose((1, 2, 0)))
        axs[0, 0].set_title(r'Planet $t_1$', fontsize=FONTSIZE)

        gt_sem_t1 = item['y_sem_t1'].squeeze()
        axs[0, 1].imshow(gt_sem_t1.numpy(), cmap='gray')
        axs[0, 1].set_title(r'GT $t_1$', fontsize=FONTSIZE)

        y_pred_sem_t1 = torch.sigmoid(logits_sem_t1).squeeze().detach()
        axs[0, 2].imshow(y_pred_sem_t1.numpy(), cmap='gray')
        axs[0, 2].set_title(r'Pred $t_1$', fontsize=FONTSIZE)

        axs[1, 0].imshow(x_t2.numpy().transpose((1, 2, 0)))
        axs[1, 0].set_title(r'Planet $t_2$', fontsize=FONTSIZE)

        gt_sem_t2 = item['y_sem_t2'].squeeze()
        axs[1, 1].imshow(gt_sem_t2.numpy(), cmap='gray')
        axs[1, 1].set_title(r'GT $t_2$', fontsize=FONTSIZE)

        y_pred_sem_t2 = torch.sigmoid(logits_sem_t2).squeeze().detach()
        axs[1, 2].imshow(y_pred_sem_t2.numpy(), cmap='gray')
        axs[1, 2].set_title(r'Pred $t_1$', fontsize=FONTSIZE)

        for _, ax in np.ndenumerate(axs):
            ax.set_axis_off()
        plt.tight_layout()

        out_file = Path(cfg.PATHS.OUTPUT) / 'plots' / 'assessment_sem_ssl' / f'{aoi_id}.png'
        out_file.parent.mkdir(exist_ok=True)
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)


def quantitative_assessment(cfg: experiment_manager.CfgNode, run_type: str = 'validation'):
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
