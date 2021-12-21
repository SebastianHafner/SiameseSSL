from pathlib import Path
from utils import geofiles, paths
import numpy as np


def bad_data() -> dict:
    dirs = paths.load_paths()
    bad_data_file = Path(dirs.HOME) / 'bad_data' / f'spacenet7_s1s2_dataset.json'
    bad_data = geofiles.load_json(bad_data_file)
    return bad_data


def timestamps() -> dict:
    dirs = paths.load_paths()
    timestamps_file = Path(dirs.DATASET) / 'spacenet7_timestamps.json'
    # TODO: if it does not exists we should create it
    assert (timestamps_file.exists())
    timestamps = geofiles.load_json(timestamps_file)
    return timestamps


def metadata() -> dict:
    dirs = paths.load_paths()
    metadata_file = Path(dirs.DATASET) / 'metadata.json'
    # TODO: if it does not exists we should create it
    assert (metadata_file.exists())
    metadata = geofiles.load_json(metadata_file)
    return metadata


def aoi_ids() -> list:
    md = metadata()
    return sorted(md['aois'].keys())


def aoi_metadata(aoi_id: str) -> list:
    md = metadata()
    return md['aois'][aoi_id]


def metadata_index(aoi_id: str, year: int, month: int) -> int:
    md = metadata()[aoi_id]
    for i, (y, m, *_) in enumerate(md):
        if y == year and month == month:
            return i


def metadata_timestamp(aoi_id: str, year: int, month: int) -> int:
    md = metadata()[aoi_id]
    for i, ts in enumerate(md):
        y, m, *_ = ts
        if y == year and month == month:
            return ts


def date2index(date: list) -> int:
    ref_value = 2019 * 12 + 1
    year, month = date
    return year * 12 + month - ref_value


# include masked data is only
def get_timeseries(aoi_id: str) -> list:
    aoi_md = aoi_metadata(aoi_id)
    timeseries = [[y, m, mask, s1, s2] for y, m, mask, s1, s2 in aoi_md if s1 and s2 and not mask]
    return timeseries


def length_timeseries(aoi_id: str) -> int:
    ts = get_timeseries(aoi_id)
    return len(ts)


def duration_timeseries(dataset: str, aoi_id: str, include_masked_data: bool = False,
                        ignore_bad_data: bool = True) -> int:
    start_year, start_month = get_date_from_index(0, dataset, aoi_id, include_masked_data, ignore_bad_data)
    end_year, end_month = get_date_from_index(-1, dataset, aoi_id, include_masked_data, ignore_bad_data)
    d_year = end_year - start_year
    d_month = end_month - start_month
    return d_year * 12 + d_month


def get_date_from_index(index: int, aoi_id: str) -> tuple:
    ts = get_timeseries(aoi_id)
    year, month, *_ = ts[index]
    return year, month


def get_geo(aoi_id: str) -> tuple:
    dirs = paths.load_paths()
    folder = Path(dirs.DATASET) / aoi_id / 'sentinel1'
    file = [f for f in folder.glob('**/*') if f.is_file()][0]
    _, transform, crs = geofiles.read_tif(file)
    return transform, crs


def get_yx_size(aoi_id: str) -> tuple:
    md = metadata()
    return md['yx_sizes'][aoi_id]


def date2str(date: list):
    year, month, *_ = date
    return f'{year - 2000:02d}-{month:02d}'


def mask_index(aoi_id: str, year: int, month: int) -> int:
    md = metadata()['aois'][aoi_id]
    md_masked = [[y, m, mask, *_] for y, m, mask, *_ in md if mask]
    for i, (y, m, *_) in enumerate(md_masked):
        if year == y and month == m:
            return i


def has_mask(aoi_id: str, year: int, month: int) -> bool:
    md = metadata()['aois'][aoi_id]
    for y, m, mask, *_ in md:
        if year == y and m == month:
            return mask


def has_masked_timestamps(aoi_id: str) -> bool:
    ts = get_timeseries(aoi_id)
    ts_masked = [[y, m, mask, *_] for y, m, mask, *_ in ts if mask]
    return True if ts_masked else False


# if no mask exists returns false for all pixels
def load_mask(aoi_id: str, year: int, month: int) -> np.ndarray:
    if has_mask(aoi_id, year, month):
        index = mask_index(aoi_id, year, month)
        masks = load_masks(aoi_id)
        return masks[:, :, index]
    else:
        return np.zeros(shape=get_yx_size(aoi_id), dtype=np.bool)


def load_masks(aoi_id: str) -> np.ndarray:
    dirs = paths.load_paths()
    masks_file = Path(dirs.DATASET) / aoi_id / f'masks_{aoi_id}.tif'
    assert (masks_file.exists())
    masks, *_ = geofiles.read_tif(masks_file)
    return masks.astype(np.bool)


def is_fully_masked(dataset: str, aoi_id: str, year: int, month: int) -> bool:
    mask = load_mask(dataset, aoi_id, year, month)
    n_elements = np.size(mask)
    n_masked = np.sum(mask)
    # TODO: mismatch due to GEE download probabably
    if n_elements * 0.9 < n_masked:
        return True
    return False


def load_label(dataset_dir: str, aoi_id: str, year: int, month: int) -> np.ndarray:
    buildings_path = Path(dataset_dir) / aoi_id / 'buildings'
    label_file = buildings_path / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
    label, _, _ = geofiles.read_tif(label_file)
    label = np.squeeze(label > 0).astype(np.float)
    mask = load_mask(aoi_id, year, month)
    label = np.where(~mask, label, np.NaN)
    return label


def load_label_in_timeseries(aoi_id: str, index: int) -> np.ndarray:
    dates = get_timeseries(aoi_id)
    year, month, *_ = dates[index]
    label = load_label(aoi_id, year, month)
    return label


def generate_change_label(aoi_id: str) -> np.ndarray:
    # computing it for spacenet7 (change between first and last label)
    label_start = load_label_in_timeseries(aoi_id, 0)
    label_end = load_label_in_timeseries(aoi_id, -1)
    change = np.logical_and(label_start == 0, label_end == 1)
    # change = np.array(label_start != label_end)
    return change.astype(np.uint8)


def generate_train_test_split(split: float = 0.3, seed: int = 7):
    np.random.seed(seed)
    rand_numbers = np.random.rand(len(aoi_ids()))
    test = rand_numbers <= split
    train = rand_numbers > split
    print('--test--')
    for in_dataset, aoi_id in zip(test, aoi_ids()):
        if in_dataset:
            print(f"'{aoi_id}',")
    print('--training--')
    for in_dataset, aoi_id in zip(train, aoi_ids()):
        if in_dataset:
            print(f"'{aoi_id}',")


if __name__ == '__main__':
    generate_train_test_split()

