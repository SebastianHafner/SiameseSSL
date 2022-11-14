import torch
from pathlib import Path
from abc import abstractmethod
import numpy as np
import multiprocessing
from utils import augmentations, experiment_manager, geofiles, spacenet7_helpers


class AbstractSpaceNet7Dataset(torch.utils.data.Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str):
        super().__init__()
        self.cfg = cfg
        self.run_type = run_type
        self.root_path = Path(cfg.PATHS.DATASET)

        self.include_alpha = cfg.DATALOADER.INCLUDE_ALPHA
        if cfg.DATALOADER.SENSOR == 'planetscope':
            self.img_bands = 3 if not self.include_alpha else 4
        else:
            self.img_bands = len(cfg.DATALOADER.SENTINEL2_BANDS)

        # creating boolean feature vector to subset sentinel 2 bands
        self.s2_indices = [['B2', 'B3', 'B4', 'B8'].index(band) for band in cfg.DATALOADER.SENTINEL2_BANDS]

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def _load_planetscope_mosaic(self, aoi_id: str, dataset: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / dataset / aoi_id / 'images_masked'
        file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}.tif'
        img, _, _ = geofiles.read_tif(file)
        img = img / 255
        # 4th band (last oen) is alpha band
        if not self.include_alpha:
            img = img[:, :, :-1]
        return img.astype(np.float32)

    def _load_sentinel2_scene(self, aoi_id: str, dataset: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / dataset / aoi_id / 'sentinel2'
        file = folder / f'sentinel2_{aoi_id}_{year}_{month:02d}.tif'
        img, *_ = geofiles.read_tif(file)
        img = img[:, :, self.s2_indices]
        return img.astype(np.float32)

    def _load_satellite_image(self, aoi_id: str, dataset: str, year: int, month: int) -> np.ndarray:
        if self.cfg.DATALOADER.SENSOR == 'planetscope':
            img = self._load_planetscope_mosaic(aoi_id, dataset, year, month)
        else:
            img = self._load_sentinel2_scene(aoi_id, dataset, year, month)
        return img

    def _load_building_label(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / 'train' / aoi_id / 'labels_raster'
        file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.tif'
        label, _, _ = geofiles.read_tif(file)
        label = label > 0
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
class SpaceNet7CDDataset(AbstractSpaceNet7Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str, no_augmentations: bool = False,
                 dataset_mode: str = None, disable_multiplier: bool = False, disable_unlabeled: bool = False,
                 only_unlabeled: bool = False):
        super().__init__(cfg, run_type)

        self.dataset_mode = cfg.DATALOADER.MODE if dataset_mode is None else dataset_mode
        self.include_building_labels = cfg.DATALOADER.INCLUDE_BUILDING_LABELS

        # handling transformations of data
        self.no_augmentations = no_augmentations
        self.transform = augmentations.compose_transformations(cfg, no_augmentations)

        self.metadata = geofiles.load_json(self.root_path / f'metadata_siamesessl.json')

        # loading labeled samples (sn7 train set) and subset to run type aoi ids
        if not only_unlabeled:
            if run_type == 'training':
                self.aoi_ids = list(cfg.DATASET.TRAIN_IDS)
            elif run_type == 'validation':
                self.aoi_ids = list(cfg.DATASET.VALIDATION_IDS)
            else:
                self.aoi_ids = list(cfg.DATASET.TEST_IDS)
        else:
            self.aoi_ids = []
        self.labeled = [True] * len(self.aoi_ids)

        # unlabeled data for semi-supervised learning
        if (cfg.DATALOADER.INCLUDE_UNLABELED and not disable_unlabeled) or only_unlabeled:
            aoi_ids_unlabelled = list(cfg.DATASET.UNLABELED_IDS)
            if cfg.DATALOADER.INCLUDE_UNLABELED_VALIDATION:
                aoi_ids_unlabelled += list(cfg.DATASET.VALIDATION_IDS)
            aoi_ids_unlabelled = sorted(aoi_ids_unlabelled)
            self.aoi_ids.extend(aoi_ids_unlabelled)
            self.labeled.extend([False] * len(aoi_ids_unlabelled))

        if not disable_multiplier:
            self.aoi_ids = self.aoi_ids * cfg.DATALOADER.TRAINING_MULTIPLIER
            self.labeled = self.labeled * cfg.DATALOADER.TRAINING_MULTIPLIER

        manager = multiprocessing.Manager()
        self.unlabeled_ids = manager.list(list(self.cfg.DATASET.UNLABELED_IDS))
        self.aoi_ids = manager.list(self.aoi_ids)
        self.labeled = manager.list(self.labeled)
        self.metadata = manager.dict(self.metadata)

        self.length = len(self.aoi_ids)

    def __getitem__(self, index):

        aoi_id = self.aoi_ids[index]
        labeled = self.labeled[index]

        timestamps = self.metadata[aoi_id]
        if self.cfg.DATALOADER.SENSOR == 'sentinel2' or self.cfg.DATALOADER.CONSISTENT_TIMESTAMPS:
            timestamps = [ts for ts in timestamps if not ts['mask'] and ts['sentinel2']]
        else:
            timestamps = [ts for ts in timestamps if not ts['mask']]

        if self.dataset_mode == 'first_last':
            i_t1, i_t2 = 0, -1
        else:
            i_t1, i_t2 = sorted(np.random.randint(0, len(timestamps), size=2))
        ts_t1, ts_t2 = timestamps[i_t1], timestamps[i_t2]

        dataset, year_t1, month_t1 = ts_t1['dataset'], ts_t1['year'], ts_t1['month']
        dataset, year_t2, month_t2 = ts_t2['dataset'], ts_t2['year'], ts_t2['month']

        img_t1 = self._load_satellite_image(aoi_id, dataset, year_t1, month_t1)
        img_t2 = self._load_satellite_image(aoi_id, dataset, year_t2, month_t2)
        imgs = np.concatenate((img_t1, img_t2), axis=-1)

        if labeled:
            change = self._load_change_label(aoi_id, year_t1, month_t1, year_t2, month_t2)
            if self.include_building_labels:
                buildings_t1 = self._load_building_label(aoi_id, year_t1, month_t1)
                buildings_t2 = self._load_building_label(aoi_id, year_t2, month_t2)
                buildings = np.concatenate((buildings_t1, buildings_t2), axis=-1).astype(np.float32)
            else:
                buildings = np.zeros((change.shape[0], change.shape[1], 2), dtype=np.float32)
        else:
            change = np.zeros((img_t1.shape[0], img_t1.shape[1], 1), dtype=np.float32)
            buildings = np.zeros((change.shape[0], change.shape[1], 2), dtype=np.float32)

        imgs, buildings, change = self.transform((imgs, buildings, change))
        img_t1, img_t2 = imgs[:self.img_bands, ], imgs[self.img_bands:, ]

        item = {
            'x_t1': img_t1,
            'x_t2': img_t2,
            'y_change': change,
            'aoi_id': aoi_id,
            'year_t1': year_t1,
            'month_t1': month_t1,
            'year_t2': year_t2,
            'month_t2': month_t2,
            'is_labeled': labeled,
        }

        if self.include_building_labels:
            buildings_t1, buildings_t2 = buildings[0, ], buildings[1, ]
            item['y_sem_t1'] = buildings_t1.unsqueeze(0)
            item['y_sem_t2'] = buildings_t2.unsqueeze(0)

        return item

    def get_index(self, aoi_id: str) -> int:
        for index, candidate_aoi_id in enumerate(self.aoi_ids):
            if aoi_id == candidate_aoi_id:
                return index
        return None

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'
