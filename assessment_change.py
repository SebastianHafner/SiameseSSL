import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from utils import experiment_manager, networks, datasets, evaluation


def qualitative_assessment(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    print(cfg.NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()
    ds = datasets.SpaceNet7CDDataset(cfg, run_type, dataset_mode='first_last', no_augmentations=True,
                                     disable_unlabeled=True, disable_multiplier=True)
    for item in ds:
        aoi_id = item['aoi_id']
        x_t1 = item['x_t1']
        x_t2 = item['x_t2']
        logits_change, logits_sem_t1, logits_sem_t2 = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))
        y_pred_change = torch.sigmoid(logits_change).squeeze().detach()
        y_pred_sem_t1 = torch.sigmoid(logits_sem_t1).squeeze().detach()
        y_pred_sem_t2 = torch.sigmoid(logits_sem_t2).squeeze().detach()

        gt_change = item['y_change'].squeeze()
        gt_sem_t1 = item['y_sem_t1'].squeeze()
        gt_sem_t2 = item['y_sem_t2'].squeeze()

        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        axs[0, 0].imshow(x_t1.numpy().transpose((1, 2, 0)))
        axs[1, 0].imshow(x_t2.numpy().transpose((1, 2, 0)))

        axs[0, 1].imshow(gt_sem_t1.numpy(), cmap='gray')
        axs[1, 1].imshow(gt_sem_t2.numpy(), cmap='gray')

        axs[0, 2].imshow(y_pred_sem_t1.numpy(), cmap='gray')
        axs[1, 2].imshow(y_pred_sem_t2.numpy(), cmap='gray')

        axs[0, 3].imshow(gt_change.numpy(), cmap='gray')
        axs[1, 3].imshow(y_pred_change.numpy(), cmap='gray')

        for _, ax in np.ndenumerate(axs):
            ax.set_axis_off()

        out_file = Path(cfg.PATHS.OUTPUT) / 'plots' / 'change' / cfg.NAME / f'{aoi_id}.png'
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)


def quantitative_assessment(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    print(cfg.NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)

    ds = datasets.SpaceNet7CDDataset(cfg, run_type, no_augmentations=True, dataset_mode='first_last',
                                     disable_multiplier=True, disable_unlabeled=True)

    data = evaluation.inference_loop(net, ds, device, False)
    f1, precision, recall = data['change']
    print(f'F1 score: {f1:.3f} - Precision: {precision:.3f} - Recall {recall:.3f}')


def assessment_argument_parser():
    # https://docs.python.org/3/library/argparse.html#the-add-argument-method
    parser = argparse.ArgumentParser(description="Experiment Args")

    parser.add_argument('-c', "--config-file", dest='config_file', required=True, help="path to config file")
    parser.add_argument('-o', "--output-dir", dest='output_dir', required=True, help="path to output directory")
    parser.add_argument('-d', "--dataset-dir", dest='dataset_dir', default="", required=True,
                        help="path to output directory")
    parser.add_argument('-r', "--run-type", dest='run_type', default="test", required=False, help="run type")

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
    quantitative_assessment(cfg, run_type=args.run_type)
