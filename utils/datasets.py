import torch
from pathlib import Path
from abc import abstractmethod
import numpy as np
from utils import augmentations, experiment_manager, geofiles, spacenet7_helpers


class AbstractSpaceNet7Dataset(torch.utils.data.Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str):
        super().__init__()
        self.cfg = cfg
        self.run_type = run_type
        self.root_path = Path(cfg.PATHS.DATASET)

        self.include_alpha = cfg.DATALOADER.INCLUDE_ALPHA
        self.img_bands = 3 if not self.include_alpha else 4

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def _load_mosaic(self, aoi_id: str, dataset: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / dataset / aoi_id / 'images_masked'
        file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}.tif'
        img, _, _ = geofiles.read_tif(file)
        img = img / 255
        # 4th band (last oen) is alpha band
        if not self.include_alpha:
            img = img[:, :, :-1]
        return img.astype(np.float32)

    def _load_building_label(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / 'train' / aoi_id / 'labels_raster'
        file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.tif'
        label, _, _ = geofiles.read_tif(file)
        label = label / 255
        return label.astype(np.float32)

    def _load_change_label(self, aoi_id: str, year_t1: int, month_t1: int, year_t2: int, month_t2) -> np.ndarray:
        building_t1 = self._load_building_label(aoi_id, year_t1, month_t1)
        building_t2 = self._load_building_label(aoi_id, year_t2, month_t2)
        change = np.logical_and(building_t1 == 0, building_t2 == 1)
        return change.astype(np.float32)

    def _load_mask(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / 'train' / aoi_id / 'labels_raster'
        file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_mask.tif'
        mask, _, _ = geofiles.read_tif(file)
        return mask.astype(np.int8)

    def get_aoi_ids(self) -> list:
        return list(set([s['aoi_id'] for s in self.samples]))

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'


# dataset for urban extraction with building footprints
class SpaceNet7SSDataset(AbstractSpaceNet7Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str, no_augmentations: bool = False):
        super().__init__(cfg, run_type)

        # handling transformations of data
        self.no_augmentations = no_augmentations
        self.transform = augmentations.compose_transformations(cfg, no_augmentations)

        # loading labeled samples (sn7 train set) and subset to run type aoi ids
        self.aoi_ids = cfg.DATASET.TRAINING_IDS if run_type == 'train' else cfg.DATASET.TEST_IDS
        data_file = self.root_path / f'metadata_train.json'
        data = geofiles.load_json(data_file)
        self.samples = []
        for aoi_id in self.aoi_ids:
            self.samples.extend(data[aoi_id])

        self.length = len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        aoi_id = sample['aoi_id']
        year = sample['year']
        month = sample['month']
        is_masked = sample['mask']
        is_labeled = sample['label']

        img = self._load_mosaic(aoi_id, 'train', year, month)

        if is_labeled:
            label = self._load_label(aoi_id, year, month)
        else:
            label = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)

        if is_masked:
            mask = self._load_mask(aoi_id, year, month)
        else:
            mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.int8)

        input_features, label = self.transform((img, label, mask))

        item = {
            'x': input_features,
            'y': label,
            'mask': mask,
            'is_masked': is_masked,
            'is_labeled': is_labeled,
            'aoi_id': aoi_id,
            'year': year,
            'month': month,
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'


# dataset for urban extraction with building footprints
class SpaceNet7CDDataset(AbstractSpaceNet7Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str, no_augmentations: bool = False,
                 dataset_mode: str = None, disable_multiplier: bool = False):
        super().__init__(cfg, run_type)

        self.dataset_mode = cfg.DATALOADER.MODE if dataset_mode is None else dataset_mode

        # handling transformations of data
        self.no_augmentations = no_augmentations
        self.transform = augmentations.compose_transformations(cfg, no_augmentations)

        # loading labeled samples (sn7 train set) and subset to run type aoi ids
        self.aoi_ids = cfg.DATASET.TRAINING_IDS if run_type == 'training' else cfg.DATASET.VALIDATION_IDS
        if not disable_multiplier:
            self.aoi_ids = self.aoi_ids * cfg.DATALOADER.TRAINING_SITES_MULTIPLIER

        self.metadata = geofiles.load_json(self.root_path / f'metadata_train.json')

        self.length = len(self.aoi_ids)

    def __getitem__(self, index):

        aoi_id = self.aoi_ids[index]

        timestamps = self.metadata[aoi_id]
        # TODO: make this work for masked data
        timestamps = [ts for ts in timestamps if not ts['mask']]

        if self.dataset_mode == 'first_last':
            indices = [0, -1]
        else:
            indices = sorted(np.random.randint(0, len(timestamps), size=2))

        year_t1, month_t1 = timestamps[indices[0]]['year'], timestamps[indices[0]]['month']
        year_t2, month_t2 = timestamps[indices[1]]['year'], timestamps[indices[1]]['month']

        img_t1 = self._load_mosaic(aoi_id, 'train', year_t1, month_t1)
        img_t2 = self._load_mosaic(aoi_id, 'train', year_t2, month_t2)

        change = self._load_change_label(aoi_id, year_t1, month_t1, year_t2, month_t2)

        img_t1, img_t2, change = self.transform((img_t1, img_t2, change))

        item = {
            'x_t1': img_t1,
            'x_t2': img_t2,
            'y': change,
            'aoi_id': aoi_id,
            'year_t1': year_t1,
            'month_t1': month_t1,
            'year_t2': year_t2,
            'month_t2': month_t2,
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'
