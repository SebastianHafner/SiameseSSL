import matplotlib.pyplot as plt
from utils import datasets, experiment_manager
import torch


def sanity_check_dataset(config_name: str, run_type: str = 'training', n_samples: int = 5):

    s2_bands = [6, 2, 1]
    s1_band = 0

    cfg = experiment_manager.load_cfg(config_name)
    ds = datasets.SpaceNet7S1S2Dataset(cfg, run_type, no_augmentations=False, disable_multiplier=True)
    for i, index in enumerate(range(len(ds))):
        item = ds.__getitem__(index)
        x = item['x'].cpu()
        s1_t1, s1_t2, s2_t1, s2_t2 = ds.split_item_x(x)

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        axs[0, 0].imshow(s2_t1[:, :, s2_bands], vmin=0, vmax=0.3)
        axs[0, 1].imshow(s2_t2[:, :, s2_bands], vmin=0, vmax=0.3)

        axs[1, 0].imshow(s1_t1[:, :, s1_band], vmin=0, vmax=1, cmap='gray')
        axs[1, 1].imshow(s1_t2[:, :, s1_band], vmin=0, vmax=1, cmap='gray')

        y = item['y'].cpu().numpy().squeeze()
        axs[0, 2].imshow(y, vmin=0, vmax=1, cmap='gray')
        axs[1, 2].set_axis_off()

        plt.show()
        plt.close(fig)

        if i >= n_samples:
            break


if __name__ == '__main__':
    sanity_check_dataset('dda_debug')
