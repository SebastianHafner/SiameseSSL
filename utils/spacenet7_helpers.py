from pathlib import Path
import numpy as np
from utils import geofiles


def file2date(file: Path) -> tuple:
    name_parts = file.stem.split('_')
    year, month = int(name_parts[2]), int(name_parts[3])
    return year, month


def get_all_aoi_ids(spacenet7_path: str, dataset: str = 'train') -> list:
    parent = Path(spacenet7_path) / dataset
    return sorted([f.name for f in parent.iterdir() if f.is_dir()])


def get_dataset(spacenet7_path: str, aoi_id: str) -> str:
    return 'train' if aoi_id in get_all_aoi_ids(spacenet7_path, 'train') else 'test'


def get_all_dates(spacenet7_path: str, aoi_id: str, sort_by_date: bool = True) -> list:
    folder = Path(spacenet7_path) / get_dataset(spacenet7_path, aoi_id) / aoi_id / 'images_masked'
    dates = [file2date(f) for f in folder.glob('**/*')]
    # removing masked
    dates = [(year, month) for year, month in dates if not is_masked(spacenet7_path, aoi_id, year, month)]
    if sort_by_date:
        dates.sort(key=lambda d: d[0] * 12 + d[1])
    return dates


def get_date_from_index(spacenet7_path: str, aoi_id: str, index: int) -> tuple:
    ts = get_all_dates(spacenet7_path, aoi_id, sort_by_date=True)
    year, month = ts[index]
    return year, month


def is_masked(spacenet7_path: str, aoi_id: str, year: int, month: int) -> bool:
    folder = Path(spacenet7_path) / get_dataset(spacenet7_path, aoi_id) / aoi_id / 'UDM_masks'
    mask_file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_UDM.tif'
    return mask_file.exists()


def load_planet_mosaic(spacenet7_path: str, aoi_id: str, year: int, month: int) -> np.ndarray:
    folder = Path(spacenet7_path) / get_dataset(spacenet7_path, aoi_id) / aoi_id / 'images_masked'
    file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}.tif'
    img, _, _ = geofiles.read_tif(file)
    return img


def get_shape(spacenet7_path: str, aoi_id: str) -> tuple:
    year, month = get_date_from_index(spacenet7_path, aoi_id, 0)
    img = load_planet_mosaic(spacenet7_path, aoi_id, year, month)
    return img.shape[0], img.shape[1]


def load_semantics_label(spacenet7_path: str, aoi_id: str, year: int, month: int) -> np.ndarray:
    folder = Path(spacenet7_path) / get_dataset(spacenet7_path, aoi_id) / aoi_id / 'labels_raster'
    label_file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.tif'
    label, _, _ = geofiles.read_tif(label_file)
    label = label > 0
    return label[:, :, 0].astype(np.float32)


def compute_change(semantics_t1: np.ndarray, semantics_t2: np.ndarray):
    change = np.logical_and(semantics_t1 == 0, semantics_t2 == 1)
    return change.astype(np.float32)


def load_change_label_dates(spacenet7_path: str, aoi_id: str, year_t1: int, month_t1: int, year_t2: int,
                            month_t2: int) -> np.ndarray:
    semantics_t1 = load_semantics_label(spacenet7_path, aoi_id, year_t1, month_t1)
    semantics_t2 = load_semantics_label(spacenet7_path, aoi_id, year_t2, month_t2)
    change = compute_change(semantics_t1, semantics_t2)
    return change


def load_change_label_indices(spacenet7_path: str, aoi_id: str, index_t1: int, index_t2: int) -> np.ndarray:
    year_t1, month_t1 = get_date_from_index(spacenet7_path, aoi_id, index_t1)
    year_t2, month_t2 = get_date_from_index(spacenet7_path, aoi_id, index_t2)
    change = load_change_label_dates(spacenet7_path, aoi_id, year_t1, month_t1, year_t2, month_t2)
    return change


def print_training_validation_split(spacenet7_path: str, split: float = 0.3, seed: int = 7):
    aoi_ids = get_all_aoi_ids(spacenet7_path, 'train')
    np.random.seed(seed)
    rand_numbers = np.random.rand(len(aoi_ids))
    validation= rand_numbers <= split
    training = rand_numbers > split
    print('--validation--')
    for in_dataset, aoi_id in zip(validation, aoi_ids):
        if in_dataset:
            print(f"'{aoi_id}',")
    print('--training--')
    for in_dataset, aoi_id in zip(training, aoi_ids):
        if in_dataset:
            print(f"'{aoi_id}',")


def print_test_aoi_ids(spacenet7_path: str):
    aoi_ids = get_all_aoi_ids(spacenet7_path, 'test')
    print('--test--')
    for aoi_id in aoi_ids:
        print(f"'{aoi_id}',")

if __name__ == '__main__':
    print_training_validation_split('C:/Users/shafner/datasets/spacenet7')
    print_test_aoi_ids('C:/Users/shafner/datasets/spacenet7')
