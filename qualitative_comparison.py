import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from utils import experiment_manager, networks, datasets

FONTSIZE = 16


def plot_misclassifications(ax, pred: np.ndarray, gt: np.ndarray):
    tp = np.logical_and(pred, gt)
    fp = np.logical_and(pred, ~gt)
    fn = np.logical_and(~pred, gt)
    classification = np.zeros(pred.shape, dtype=np.uint8)
    classification[tp] = 1
    classification[fp] = 2
    classification[fn] = 3
    pass


def qualitative_comparison(config_names: list, output_dir: str, dataset_dir: str, aoi_ids: list):
    plot_size = 3
    rows = len(aoi_ids)
    cols = 3 + len(config_names)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * plot_size, rows * plot_size))
    for _, ax in np.ndenumerate(axs):
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()

    for j, config_name in enumerate(config_names):
        print(config_name)
        cfg = experiment_manager.setup_cfg_manual(config_name, Path(output_dir), Path(dataset_dir))
        net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, 'cpu')
        net.eval()
        ds = datasets.SpaceNet7CDDataset(cfg, 'test', dataset_mode='first_last', no_augmentations=True,
                                         disable_unlabeled=True, disable_multiplier=True)
        for i, aoi_id in enumerate(tqdm(aoi_ids)):

            index = ds.get_index(aoi_id)
            item = ds.__getitem__(index)
            x_t1 = item['x_t1']
            x_t2 = item['x_t2']

            logits_change, *_ = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))
            y_prob_change = torch.sigmoid(logits_change).squeeze().detach().numpy()
            y_pred_change = (y_prob_change > 0.5).astype(np.uint8)
            axs[i, 3 + j].imshow(y_pred_change, cmap='gray')

            if j == 0:
                axs[i, 0].imshow(x_t1.numpy().transpose((1, 2, 0)))
                axs[i, 1].imshow(x_t2.numpy().transpose((1, 2, 0)))
                gt_change = item['y_change'].squeeze()
                axs[i, 2].imshow(gt_change.numpy(), cmap='gray')

    for c in range(cols):
        char = chr(97 + c)
        axs[-1, c].set_xlabel(f'({char})', fontsize=FONTSIZE, fontweight='bold')
        axs[-1, c].xaxis.set_label_coords(0.5, -0.025)

    out_file = Path(output_dir) / 'plots' / f'qualitative_comparison.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def assessment_argument_parser():
    # https://docs.python.org/3/library/argparse.html#the-add-argument-method
    parser = argparse.ArgumentParser(description="Experiment Args")

    parser.add_argument('-c', '--config-files', nargs='+', required=True, help="path to config file")
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
    aoi_ids = [
        'L15-0357E-1223N_1429_3296_13',
        'L15-0457E-1135N_1831_3648_13',
        'L15-0566E-1185N_2265_3451_13',
        'L15-0571E-1075N_2287_3888_13',
        'L15-1209E-1113N_4838_3737_13',
        'L15-1210E-1025N_4840_4088_13',
        'L15-1276E-1107N_5105_3761_13',
        'L15-1479E-1101N_5916_3785_13',
        'L15-1669E-1160N_6678_3548_13',
        'L15-1672E-1207N_6691_3363_13',
        'L15-1690E-1211N_6763_3346_13',
        'L15-1703E-1219N_6813_3313_13',
    ]

    aoi_ids = [
        'L15-0457E-1135N_1831_3648_13',
        'L15-0566E-1185N_2265_3451_13',
        'L15-1479E-1101N_5916_3785_13',
        'L15-1703E-1219N_6813_3313_13',
    ]
    qualitative_comparison(args.config_files, args.output_dir, args.dataset_dir, aoi_ids)
    # qualitative_assessment_change(cfg, run_type=args.run_type)
    # qualitative_assessment_sem(cfg, run_type=args.run_type)
