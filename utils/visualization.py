import matplotlib.pyplot as plt
import matplotlib.colors as colors
from utils.geofiles import *
import numpy as np
from pathlib import Path


def plot_optical(ax, img: np.ndarray, scale_factor: float = 0.4, show_title: bool = False):
    band_indices = [0, 1, 2]
    img = img[:, :, band_indices]
    img = img.clip(0, 1)
    ax.imshow(img, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    if show_title:
        ax.set_title('s2 mosaic')


def plot_probability(ax, probability: np.ndarray, show_title: bool = False):
    ax.imshow(probability, cmap='jet', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    if show_title:
        ax.set_title('activation')


def plot_buildings(ax, prediction: np.ndarray, show_title: bool = False):
    cmap = colors.ListedColormap(['#ffffff', '#b00000'])
    boundaries = [0, 0.5, 1]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    ax.imshow(prediction, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    if show_title:
        ax.set_title('prediction')


def plot_polygons(ax, polygons: list, show_title: bool = False):

    for polygon in polygons:
        ax.plot(*polygon.exterior.xy)

    ax.set_xticks([])
    ax.set_yticks([])
    if show_title:
        ax.set_title('prediction')
