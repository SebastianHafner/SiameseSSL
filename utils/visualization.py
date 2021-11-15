import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from utils import geofiles, dataset_helpers, paths


def plot_optical(ax, aoi_id: str, year: int, month: int, vis: str = 'true_color',
                 rescale_factor: float = 0.4):
    ax.set_xticks([])
    ax.set_yticks([])
    dirs = paths.load_paths()
    file = Path(dirs.DATASET) / aoi_id / 'sentinel2' / f'sentinel2_{aoi_id}_{year}_{month:02d}.tif'
    if not file.exists():
        return
    img, _, _ = geofiles.read_tif(file)
    band_indices = [2, 1, 0] if vis == 'true_color' else [6, 2, 1]
    bands = img[:, :, band_indices] / rescale_factor
    bands = bands.clip(0, 1)
    ax.imshow(bands)


def plot_sar(ax, aoi_id: str, year: int, month: int, vis: str = 'VV'):
    ax.set_xticks([])
    ax.set_yticks([])
    dirs = paths.load_paths()
    file = Path(dirs.DATASET) / aoi_id / 'sentinel1' / f'sentinel1_{aoi_id}_{year}_{month:02d}.tif'
    if not file.exists():
        return
    img, _, _ = geofiles.read_tif(file)
    band_index = 0 if vis == 'VV' else 1
    bands = img[:, :, band_index]
    bands = bands.clip(0, 1)
    ax.imshow(bands, cmap='gray')


def plot_buildings(ax, aoi_id: str, year: int, month: int):
    buildings = label_helpers.load_label(aoi_id, year, month)
    isnan = np.isnan(buildings)
    buildings = buildings.astype(np.uint8)
    buildings = np.where(~isnan, buildings, 3)
    colors = [(0, 0, 0), (1, 1, 1), (1, 0, 0)]
    cmap = mpl.colors.ListedColormap(colors)
    ax.imshow(buildings, cmap=cmap, vmin=0, vmax=2)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_change_label(ax, aoi_id: str, year_t1: int, month_t1: int, year_t2: int, month_t2: int):
    label_start = dataset_helpers.load_label(aoi_id, year_t1, month_t1)
    label_end = dataset_helpers.load_label(aoi_id, year_t2, month_t2)
    change = np.array(label_start != label_end)
    return change.astype(np.float32)


def plot_blackwhite(ax, img: np.ndarray, cmap: str = 'gray'):
    ax.imshow(img.clip(0, 1), cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_classification(ax, pred: np.ndarray, dataset: str, aoi_id: str):
    label = label_helpers.generate_change_label(dataset, aoi_id, config.include_masked()).astype(np.bool)
    pred = pred.squeeze().astype(np.bool)
    tp = np.logical_and(pred, label)
    fp = np.logical_and(pred, ~label)
    fn = np.logical_and(~pred, label)

    img = np.zeros(pred.shape, dtype=np.uint8)

    img[tp] = 1
    img[fp] = 2
    img[fn] = 3

    colors = [(0, 0, 0), (1, 1, 1), (142 / 255, 1, 0), (140 / 255, 25 / 255, 140 / 255)]
    cmap = mpl.colors.ListedColormap(colors)
    ax.imshow(img, cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_mask(ax, dataset: str, aoi_id: str, year: int, month: int):
    mask = dataset_helpers.load_mask(dataset, aoi_id, year, month)
    ax.imshow(mask.astype(np.uint8), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])

