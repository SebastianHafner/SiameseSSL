import torch
from torchvision import transforms
import json
from pathlib import Path

from utils.augmentations import *
from utils.geofiles import *


# dataset for urban extraction with building footprints
class SpaceNet7Dataset(torch.utils.data.Dataset):

    def __init__(self, cfg, dataset: str, split: str = None, no_augmentations: bool = False, single_aoi: str = None,
                 sort_by_date: bool = False):
        super().__init__()

        # assigning class variables
        self.cfg = cfg
        self.dataset = dataset
        self.root_path = Path(cfg.DATASETS.PATH)

        self.include_alpha = cfg.DATALOADER.INCLUDE_ALPHA
        self.img_bands = 3 if not self.include_alpha else 4
        self.upsample = cfg.DATALOADER.UPSAMPLE

        # handling transformations of data
        self.no_augmentations = no_augmentations
        self.padding = PadToEven()
        self.transform = compose_transformations(cfg, no_augmentations)

        # loading samples and applying split if passed
        data_file = self.root_path / f'{dataset}_samples.json'
        data = load_json(data_file)
        self.aoi_dates = data['aoi_dates']
        all_samples = data['samples']
        self.samples = all_samples if split is None else [s for s in all_samples if s['split'] == split]
        if single_aoi is not None:
            self.samples = [s for s in self.samples if s['aoi_id'] == single_aoi]
        if sort_by_date:
            # TODO: I broke this one and I'm gonna fix it
            self._sort_by_date()
        self.length = len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        aoi_id = sample['aoi_id']
        year = sample['year']
        month = sample['month']
        ts_index = sample['ts_index']

        # stacking all images
        offsets = self.cfg.DATALOADER.IMAGE_STACK_OFFSETS
        # TODO: don't hard code image shapes
        img_height, img_width = 1024, 1024
        input_features = np.empty((img_height, img_width, len(offsets) * self.img_bands), dtype=np.float32)
        dates = self.aoi_dates[aoi_id]
        for i, offset in enumerate(offsets):
            ts_index_offset = ts_index + offset
            if 0 <= ts_index_offset < len(dates):
                year_offset, month_offset = self.aoi_dates[aoi_id][ts_index_offset]
                img = self._load_mosaic(aoi_id, year_offset, month_offset)
            else:
                img = np.zeros((img_height, img_width, self.img_bands), dtype=np.float32)
            input_features[:, :, i * self.img_bands: (i + 1) * self.img_bands] = img

        if self.dataset == 'train':
            label = self._load_label(aoi_id, year, month)
        else:
            # TODO: fix shape for test set
            label = np.zeros((img_height, img_width, 1), dtype=np.float32)

        input_features, label = self.transform((input_features, label))

        item = {
            'x': input_features,
            'y': label,
            'aoi_id': aoi_id,
            'year': year,
            'month': month,
        }

        return item

    def _load_mosaic(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / self.dataset / aoi_id / 'images_masked'
        file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}.tif'
        img, _, _ = read_tif(file)
        img = img / 255
        # 4th band (last oen) is alpha band
        if not self.include_alpha:
            img = img[:, :, :-1]
        img = PadToEven.pad_to_even(img)
        return img.astype(np.float32)

    def _load_label(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        if self.upsample:
            folder = self.root_path / self.dataset / aoi_id / f'labels_raster_X{self.upsample}'
        else:
            folder = self.root_path / self.dataset / aoi_id / 'labels_raster'
        file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.tif'
        label, _, _ = read_tif(file)
        label = label / 255
        label = PadToEven.pad_to_even(label)
        return label.astype(np.float32)

    def get_aoi_ids(self) -> list:
        return list(set([s['aoi_id'] for s in self.samples]))

    def _sort_by_date(self):
        sorted_samples = []
        for aoi_id in self.get_aoi_ids():
            # (sample, timestamp)
            aoi_samples = [s for s in self.samples if s['aoi_id'] == aoi_id]
            timestamps = [int(int(s['year']) * 12 + int(s['month'])) for s in aoi_samples]
            sorted_samples.extend([s for _, s in sorted(zip(timestamps, aoi_samples))])
        assert(len(self.samples) == len(sorted_samples))
        self.samples = sorted_samples

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'
