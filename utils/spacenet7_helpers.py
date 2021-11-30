from pathlib import Path
import numpy as np


def file2date(file: Path) -> tuple:
    name_parts = file.stem.split('_')
    year, month = int(name_parts[2]), int(name_parts[3])
    return year, month


def get_all_aoi_ids(spacenet7_path: str, dataset: str = 'train') -> list:
    parent = Path(spacenet7_path) / dataset
    return sorted([f.name for f in parent.iterdir() if f.is_dir()])


def get_all_dates(spacenet7_path: str, dataset: str, aoi_id: str, sort_by_date: bool = True) -> list:
    folder = Path(spacenet7_path) / dataset / aoi_id / 'images_masked'
    dates = [file2date(f) for f in folder.glob('**/*')]
    if sort_by_date:
        dates.sort(key=lambda d: d[0] * 12 + d[1])
    return dates


def is_masked(spacenet7_path: str, dataset: str, aoi_id: str, year: int, month: int) -> bool:
    folder = Path(spacenet7_path) / dataset / aoi_id / 'UDM_masks'
    mask_file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_UDM.tif'
    return mask_file.exists()


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
