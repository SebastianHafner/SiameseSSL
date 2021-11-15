from pathlib import Path
from networks.network_loader import load_network
from utils.dataloader import SpaceNet7Dataset
from experiment_manager.config import config
from utils.metrics import *
from tqdm import tqdm
from torch.utils import data as torch_data
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.geofiles import *
from utils.visualization import *

DATASET_PATH = Path('/storage/shafner/spacenet7')
CONFIG_PATH = Path('/home/shafner/spacenet7/configs')
NETWORK_PATH = Path('/storage/shafner/spacenet7/networks/')


def random_selection(config_name: str, checkpoint: int, n: int):
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')

    # loading dataset
    dataset = SpaceNet7Dataset(cfg, dataset='test', no_augmentations=True)

    # loading network
    net = load_network(cfg, NETWORK_PATH / f'{config_name}_{checkpoint}.pkl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    item_indices = list(np.random.randint(0, len(dataset), size=n))

    for index in item_indices:
        sample = dataset.__getitem__(index)
        aoi_id = sample['aoi_id']
        aoi_date = sample['aoi_date']

        fig, axs = plt.subplots(1, 3, figsize=(8, 4))
        fig.suptitle(f'{run_type} {aoi_id}')

        optical_file = DATASET_PATH / 'test' / aoi_id / 'images_masked' / f'{aoi_date}.tif'
        plot_optical(axs[0], optical_file, show_title=True)

        with torch.no_grad():
            x = sample['x'].to(device)
            x = x * 255
            logits = net(x.unsqueeze(0))
            prob = torch.sigmoid(logits[0, 0,])
            prob = prob.detach().cpu().numpy()
            pred = prob > cfg.THRESH

            plot_probability(axs[1], prob, show_title=True)
            plot_prediction(axs[2], pred, show_title=True)

        plt.show()


def progression_monitoring(config_name: str, checkpoint: int, aoi_id: str):
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')

    # loading dataset
    dataset = SpaceNet7Dataset(cfg, dataset='test', no_augmentations=True)

    # loading network
    net = load_network(cfg, NETWORK_PATH / f'{config_name}_{checkpoint}.pkl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    item_indices = list(np.random.randint(0, len(dataset), size=n))

    for index in item_indices:
        sample = dataset.__getitem__(index)
        aoi_id = sample['aoi_id']
        aoi_date = sample['aoi_date']

        fig, axs = plt.subplots(1, 3, figsize=(6, 2))
        fig.suptitle(f'{run_type} {aoi_id}')

        optical_file = DATASET_PATH / 'test' / aoi_id / 'images_masked' / f'{aoi_date}.tif'
        plot_optical(axs[0], optical_file, show_title=True)

        with torch.no_grad():
            x = sample['x'].to(device)
            logits = net(x.unsqueeze(0))
            prob = torch.sigmoid(logits[0, 0,])
            prob = prob.detach().cpu().numpy()
            pred = prob > cfg.THRESH

            plot_probability(axs[1], prob, show_title=True)
            plot_prediction(axs[2], pred, show_title=True)

        plt.show()


if __name__ == '__main__':
    config_name = 'baseline'
    checkpoint = 100
    run_type = 'validation'

    random_selection(config_name, checkpoint, n=20)

