import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from utils import experiment_manager, networks, datasets, spacenet7_helpers

FONTSIZE = 16


def get_misclassifications(pred: np.ndarray, gt: np.ndarray):
    pred, gt = pred.astype(np.bool), gt.astype(np.bool)
    tp = np.logical_and(pred, gt)
    fp = np.logical_and(pred, ~gt)
    fn = np.logical_and(~pred, gt)
    classification = np.zeros(pred.shape, dtype=np.uint8)
    classification[tp] = 1
    classification[fp] = 2
    classification[fn] = 3
    return classification


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


def qualitative_comparison_zoom_selector(output_dir: str, dataset_dir: str, aoi_ids: list, zooms: list):
    plot_size = 3
    rows = len(aoi_ids)
    cols = 5
    fig, axs = plt.subplots(rows, cols, figsize=(cols * plot_size, rows * plot_size))
    plt.tight_layout()

    for i, (aoi_id, zoom) in enumerate(zip(aoi_ids, zooms)):
        i_start, j_start, s = zoom
        if s is None:
            i_end, j_end = spacenet7_helpers.get_shape(dataset_dir, aoi_id)
        else:
            i_end, j_end = i_start + s, j_start + s
        for index in [0, -1]:
            year, month = spacenet7_helpers.get_date_from_index(dataset_dir, aoi_id, index)
            img = spacenet7_helpers.load_planet_mosaic(dataset_dir, aoi_id, year, month)
            img = img[i_start:i_end, j_start:j_end, ]
            axs[i, 0 if index == 0 else 2].imshow(img)

            gt_semantics = spacenet7_helpers.load_semantics_label(dataset_dir, aoi_id, year, month)
            gt_semantics = gt_semantics[i_start:i_end, j_start:j_end]
            axs[i, 1 if index == 0 else 3].imshow(gt_semantics, cmap='gray')

        gt_change = spacenet7_helpers.load_change_label_indices(dataset_dir, aoi_id, 0, -1)
        gt_change = gt_change[i_start:i_end, j_start:j_end]
        axs[i, 4].imshow(gt_change, cmap='gray')

    out_file = Path(output_dir) / 'plots' / f'zoom_selector.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def qualitative_comparison_zoom(config_names: list, output_dir: str, dataset_dir: str, aoi_ids: list, zooms: list,
                                colored: bool = False):
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
        for i, (aoi_id, zoom) in enumerate(tqdm(zip(aoi_ids, zooms))):
            i_start, j_start, s = zoom
            if s is None:
                i_end, j_end = spacenet7_helpers.get_shape(dataset_dir, aoi_id)
            else:
                i_end, j_end = i_start + s, j_start + s

            index = ds.get_index(aoi_id)
            item = ds.__getitem__(index)
            x_t1 = item['x_t1']
            x_t2 = item['x_t2']

            logits_change, *_ = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))
            y_prob_change = torch.sigmoid(logits_change).squeeze().detach().numpy()
            y_pred_change = (y_prob_change > 0.5).astype(np.uint8)
            y_pred_change = y_pred_change[i_start:i_end, j_start:j_end]

            gt_change = spacenet7_helpers.load_change_label_indices(dataset_dir, aoi_id, 0, -1)
            gt_change = gt_change[i_start:i_end, j_start:j_end]
            if colored:
                classification = get_misclassifications(y_pred_change, gt_change)
                colors = [(0, 0, 0), (1, 1, 1), (142 / 255, 1, 0), (140 / 255, 25 / 255, 140 / 255)]
                cmap = mpl.colors.ListedColormap(colors)
                axs[i, 3 + j].imshow(classification, cmap=cmap, vmin=0, vmax=3)
            else:
                axs[i, 3 + j].imshow(y_pred_change, cmap='gray')

            if j == 0:
                for index in [0, -1]:
                    year, month = spacenet7_helpers.get_date_from_index(dataset_dir, aoi_id, index)
                    img = spacenet7_helpers.load_planet_mosaic(dataset_dir, aoi_id, year, month)
                    img = img[i_start:i_end, j_start:j_end, ]
                    axs[i, 0 if index == 0 else 1].imshow(img)
                    axs[i, 2].imshow(gt_change, cmap='gray')

    for c in range(cols):
        char = chr(97 + c)
        axs[-1, c].set_xlabel(f'({char})', fontsize=FONTSIZE, fontweight='bold')
        axs[-1, c].xaxis.set_label_coords(0.5, -0.025)

    suffix = 'zoom' if not colored else 'zoom_colored'
    out_file = Path(output_dir) / 'plots' / f'qualitative_comparison_{suffix}.png'
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
        'L15-1672E-1207N_6691_3363_13',
    ]
    zooms = [
        (400, 100, 200),
        (0, 230, 570),
        (0, 0, 300),
        (0, 150, 750),
    ]

    aoi_ids = [
        'L15-0457E-1135N_1831_3648_13',
        'L15-1479E-1101N_5916_3785_13',
        'L15-1672E-1207N_6691_3363_13',
    ]
    zooms = [
        (400, 100, 200),
        (0, 0, 300),
        (0, 150, 750),
    ]

    # qualitative_comparison(args.config_files, args.output_dir, args.dataset_dir, aoi_ids)
    # qualitative_comparison_zoom_selector(args.output_dir, args.dataset_dir, aoi_ids, zooms)
    qualitative_comparison_zoom(args.config_files, args.output_dir, args.dataset_dir, aoi_ids, zooms, colored=True)
    # qualitative_assessment_change(cfg, run_type=args.run_type)
    # qualitative_assessment_sem(cfg, run_type=args.run_type)
