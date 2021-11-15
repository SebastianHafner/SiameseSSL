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


def quantitative_validation(config_name: str, checkpoint: int, save_output: bool = False):
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')

    # loading network
    net = load_network(cfg, NETWORK_PATH / f'{config_name}_{checkpoint}.pkl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    thresh = cfg.THRESH

    output_data = {'training': {}, 'validation': {}}
    # training can be added
    for run_type in ['training', 'validation']:

        # TODO: maybe do validation per aoi id
        print(f'Quantitative assessment {run_type}')
        dataset = SpaceNet7Dataset(cfg=cfg, run_type=run_type, no_augmentations=True)

        dataloader_kwargs = {
            'batch_size': 1,
            'num_workers': cfg.DATALOADER.NUM_WORKER,
            'shuffle': False,
            'drop_last': False,
            'pin_memory': True,
        }
        dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

        y_true_set, y_pred_set = np.array([]), np.array([])
        for i, batch in enumerate(tqdm(dataloader)):
            with torch.no_grad():
                x = batch['x'].to(device)
                y_true = batch['y'].to(device)
                logits = net(x)
                y_pred = torch.sigmoid(logits) > thresh

                y_true = y_true.detach().cpu().flatten().numpy()
                y_pred = y_pred.detach().cpu().flatten().numpy()
                y_true_set = np.concatenate((y_true_set, y_true))
                y_pred_set = np.concatenate((y_pred_set, y_pred))

        y_true_set, y_pred_set = torch.Tensor(np.array(y_true_set)), torch.Tensor(np.array(y_pred_set))
        prec = precision(y_true_set, y_pred_set, dim=0)
        rec = recall(y_true_set, y_pred_set, dim=0)
        f1 = f1_score(y_true_set, y_pred_set, dim=0)

        print(f'Precision: {prec.item():.3f} - Recall: {rec.item():.3f} - F1 score: {f1.item():.3f}')

        output_data[run_type] = {'f1_score': f1.item(), 'precision': prec.item(), 'recall': rec.item()}

    if save_output:
        output_file = DATASET_PATH / 'validation' / f'validation_{config_name}.json'
        save_json(output_file, output_data)


def random_selection(config_name: str, checkpoint: int, run_type: str, n: int = None, save_output: bool = False):
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')

    # loading dataset
    dataset = SpaceNet7Dataset(cfg, dataset='train', split=run_type, no_augmentations=True)

    # loading network
    net = load_network(cfg, NETWORK_PATH / f'{config_name}_{checkpoint}.pkl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    if n is None:
        item_indices = np.arange(0, len(dataset))
    else:
        item_indices = list(np.random.randint(0, len(dataset), size=n))

    for index in item_indices:
        sample = dataset.__getitem__(index)
        aoi_id = sample['aoi_id']
        aoi_date = sample['aoi_date']
        # TODO: include month and year in sample data and remove this part
        aoi_date_parts = aoi_date.split('_')
        month = int(aoi_date_parts[3])
        year = int(aoi_date_parts[2])

        fig, axs = plt.subplots(1, 4, figsize=(10, 4))
        fig.suptitle(f'{run_type} {aoi_id}')

        optical_file = DATASET_PATH / 'train' / aoi_id / 'images_masked' / f'{aoi_date}.tif'
        plot_optical(axs[0], optical_file, show_title=True)

        label_file = DATASET_PATH / 'train' / aoi_id / 'labels_raster' / f'{aoi_date}_Buildings.tif'
        plot_buildings(axs[1], label_file, show_title=True)

        with torch.no_grad():
            x = sample['x'].to(device)
            # TODO: remove this after training a new network
            x = x * 255
            logits = net(x.unsqueeze(0))
            prob = torch.sigmoid(logits[0, 0, ])
            prob = prob.detach().cpu().numpy()
            pred = prob > cfg.THRESH

            plot_probability(axs[2], prob, show_title=True)
            plot_prediction(axs[3], pred, show_title=True)

        if save_output:
            file_name = f'building_probability_{aoi_id}_{year}_{month:02d}.npy'
            output_file = DATASET_PATH / 'validation' / file_name
            np.save(output_file, prob)
        else:
            plt.show()
        plt.close()


if __name__ == '__main__':
    config_name = 'lr-4'
    checkpoint = 100
    run_type = 'validation'

    random_selection(config_name, checkpoint, run_type=run_type, n=None, save_output=True)



