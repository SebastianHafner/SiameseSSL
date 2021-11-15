import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import experiment_manager, networks, datasets


def qual_assessment(config_name: str, run_type: str = 'validation'):
    cfg = experiment_manager.load_cfg(config_name)
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, 'cpu')
    net.eval()
    ds = datasets.SpaceNet7S1S2Dataset(cfg, run_type, 'first_last', no_augmentations=True, include_unlabeled=False,
                                       disable_multiplier=True)

    for item in ds:
        x = item['x'].unsqueeze(0)
        y_pred = net(x)
        y_pred = torch.sigmoid(y_pred).squeeze().detach().numpy()
        y_gts = item['y'].squeeze().numpy()
        _, _, s2_t1, s2_t2 = ds.split_item_x(x)

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        bands = [2, 1, 0]  # [6, 2, 1]
        scale_factor = .3
        img_s2_t1 = np.clip(s2_t1.squeeze()[bands].numpy().transpose((1, 2, 0)) / scale_factor, 0, 1)
        axs[0, 0].imshow(img_s2_t1)
        img_s2_t2 = np.clip(s2_t2.squeeze()[bands].numpy().transpose((1, 2, 0)) / scale_factor, 0, 1)
        axs[0, 1].imshow(img_s2_t2)
        axs[1, 0].imshow(y_gts, vmin=0, vmax=1, cmap='gray')
        axs[1, 1].imshow(y_pred, vmin=0, vmax=1, cmap='gray')

        for _, ax in np.ndenumerate(axs):
            ax.set_axis_off()

        plt.show()
        plt.close(fig)


def quan_assessment(config_name: str, run_type: str = 'validation'):
    cfg = experiment_manager.load_cfg(config_name)
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, 'cpu')
    net.eval()
    ds = datasets.SpaceNet7S1S2Dataset(cfg, run_type, 'first_last', no_augmentations=True, include_unlabeled=False,
                                       disable_multiplier=True)

    for item in ds:
        x = item['x'].unsqueeze(0)
        y_pred = net(x)
        y_pred = torch.sigmoid(y_pred).squeeze().detach().numpy()


def qual_assessment_stockholm(config_name: str):
    cfg = experiment_manager.load_cfg(config_name)
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, 'cpu')
    net.eval()

    ds = datasets.StockholmS1S2Dataset(cfg, from_date='2015-08', to_date='2021-04')

    with torch.no_grad():
        for i in tqdm(range(len(ds))):
            patch = ds.__getitem__(i)
            img = patch['x']
            logits = net(img.unsqueeze(0))
            prob = torch.sigmoid(logits) * 100
            prob = prob.squeeze().cpu().numpy().astype('uint8')
            prob = np.clip(prob, 0, 100)
            center_prob = prob[dataset.patch_size:dataset.patch_size * 2, dataset.patch_size:dataset.patch_size * 2]

            i_start = patch['i']
            i_end = i_start + dataset.patch_size
            j_start = patch['j']
            j_end = j_start + dataset.patch_size
            prob_output[i_start:i_end, j_start:j_end, 0] = center_prob

    output_file = save_path / f'prob_{site}_{config_name}.tif'
    write_tif(output_file, prob_output, transform, crs)
    pass


if __name__ == '__main__':
    qual_assessment('opticalunet_baseline', run_type='training')
    # qual_assessment_stockholm('opticalunet_baseline')
