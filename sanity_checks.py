import matplotlib.pyplot as plt
from utils.dataloader import SpaceNet7Dataset
import numpy as np
from experiment_manager.config import config
from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils import data as torch_data
from networks.network_loader import load_network
from utils.visualization import *
from postprocessing import segmentation2polygons

ROOT_PATH = Path('/home/shafner/spacenet7')
CONFIG_PATH = ROOT_PATH / 'configs'
NET_PATH = Path('/storage/shafner/spacenet7/networks')
PLOTS_SAVE_PATH = Path('/storage/shafner/spacenet7/plots')


def plot_samples(n: int, img_scale_factor: float = 1, seed: int = 7):

    cfg = config.load_cfg(CONFIG_PATH / 'base2.yaml')
    dataset = SpaceNet7Dataset(cfg, dataset='train', no_augmentations=True)
    np.random.seed(seed)
    indices = np.random.randint(0, len(dataset), n)

    for index in indices:
        sample = dataset.__getitem__(index)
        img = sample['x'].cpu().detach().squeeze().numpy()
        gt = sample['y'].cpu().detach().squeeze().numpy()
        aoi_id = sample['aoi_id']

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        plt.suptitle(aoi_id)
        axs[0].imshow(img.transpose((1, 2, 0)) / img_scale_factor)
        axs[1].imshow(gt, interpolation='nearest')

        for ax in axs:
            ax.set_axis_off()

        plt.show()
        plt.close()


def plot_time_series(config_name: str, checkpoint: int, aoi_id: str, save_plot: bool = False,
                     max_length: int = None):

    # loading config and network
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')
    net = load_network(cfg, NET_PATH / f'{config_name}_{checkpoint}.pkl')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    # preparing dataset
    dataset = SpaceNet7Dataset(cfg=cfg, dataset='train', single_aoi=aoi_id, sort_by_date=True, no_augmentations=True)

    # setting threshold
    if cfg.THRESH == -1:
        print('No threshold set, using 0.5')
        thresh = 0.5
    else:
        thresh = cfg.THRESH

    # creating plots for time series
    timestamps = len(dataset) if max_length is None else max_length
    fig, axs = plt.subplots(4, timestamps, figsize=(timestamps * 2, 8))
    with torch.no_grad():
        for index in tqdm(range(timestamps)):

            item = dataset.__getitem__(index)

            # probability
            img = item['x'].to(device)
            logits = net(img.unsqueeze(0))
            probs = torch.sigmoid(logits).cpu().detach().squeeze().numpy()
            plot_probability(axs[1, index], probs)

            # apply postprocessing to probs
            polygons = segmentation2polygons(np.flipud(probs), cfg)
            plot_polygons(axs[2, index], polygons)

            # label
            label = item['y']
            plot_buildings(axs[3, index], label.cpu().detach().squeeze().numpy())

            # sentinel 2
            year, month = int(item['year']), int(item['month'])
            axs[0, index].set_title(f'{year}-{month}')
            plot_optical(axs[0, index], img.cpu().detach().squeeze().numpy().transpose((1, 2, 0)))

    plt.suptitle(aoi_id)
    plt.subplots_adjust(wspace=0, hspace=0)

    if save_plot:
        output_file = PLOTS_SAVE_PATH / f'time_series_{config_name}_{aoi_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()







if __name__ == '__main__':
    # plot_samples(10)

    # validation aoi ids
    # 'L15-0368E-1245N_1474_3210_13'
    # 'L15-1716E-1211N_6864_3345_13'
    # 'L15-1439E-1134N_5759_3655_13'
    # 'L15-0487E-1246N_1950_3207_13'
    # 'L15-0566E-1185N_2265_3451_13'
    # 'L15-1709E-1112N_6838_3742_13'
    # 'L15-0577E-1243N_2309_3217_13'
    # 'L15-0387E-1276N_1549_3087_13'

    # plot_time_series('upsample_cosine', 75, 'L15-0368E-1245N_1474_3210_13', max_length=12)

    validation_aoi_ids = [
        'L15-0368E-1245N_1474_3210_13', 'L15-1716E-1211N_6864_3345_13', 'L15-1439E-1134N_5759_3655_13',
        'L15-0487E-1246N_1950_3207_13', 'L15-0566E-1185N_2265_3451_13', 'L15-1709E-1112N_6838_3742_13',
        'L15-0577E-1243N_2309_3217_13', 'L15-0387E-1276N_1549_3087_13']
    for aoi_id in validation_aoi_ids:
        plot_time_series('twoimages', 75, aoi_id, max_length=8, save_plot=True)
