from pathlib import Path
import numpy as np
import cv2
from utils.geofiles import *
from tqdm import tqdm
import matplotlib.pyplot as plt

DATASET_FOLDER = Path('/storage/shafner/spacenet7')
SEED = 42
UPSAMPLE = 2


def file2date(file: Path) -> tuple:
    name_parts = file.stem.split('_')
    year, month = int(name_parts[2]), int(name_parts[3])
    return year, month


def get_all_aoi_ids(dataset: str = 'train') -> list:
    parent = DATASET_FOLDER / dataset
    return [f.name for f in parent.iterdir() if f.is_dir()]


def get_all_dates(dataset: str, aoi_id: str, sort_by_date: bool) -> list:
    folder = DATASET_FOLDER / dataset / aoi_id / 'images_masked'
    dates = [file2date(f) for f in folder.glob('**/*')]
    if sort_by_date:
        dates.sort(key= lambda d: d[0] * 12 + d[1])
    return dates


def round_to_255(img):
    img[img>127] = 255
    img[img<=127] = 0
    return img


def create_building_masks(aoi_id: str, save: bool = False):
    print(f'creating building masks for {aoi_id}...')
    aoi_folder = DATASET_FOLDER / 'train' / aoi_id
    image_folder = aoi_folder / 'images_masked'

    image_files = sorted([f for f in image_folder.glob('**/*')])
    for i, image_file in enumerate(image_files):
        # prepare empty mask
        img, transform, crs = read_tif(image_file)
        mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        mask_upsample = np.zeros((img.shape[0]*UPSAMPLE, img.shape[1]*UPSAMPLE, 1), dtype=np.uint8)

        # load buildings polygons and fill mask
        buildings_file = aoi_folder / 'labels_match_pix' / f'{image_file.stem}_Buildings.geojson'
        feature_collection = load_json(buildings_file)
        buildings = feature_collection['features']
        for building in buildings:
            # TODO: maybe multipolygons could cause problems when just using the first poly of a building
            # list of building elements: first element is the building outline and others are holes
            building_elements = building['geometry']['coordinates']

            # filling in the whole building
            building_outline = building_elements[0]
            first_coord = building_outline[0]
            # TODO: some coords are 3-d for some stupid reason, maybe fix?
            if len(first_coord) == 3:
                building_outline = [coord[:2] for coord in building_outline]
            cv2.fillPoly(mask, [np.rint(np.array(building_outline)).astype(int)], 255, cv2.LINE_AA)
            # Repeat on the upsampled mask
            cv2.fillPoly(mask_upsample, [np.rint(np.array(building_outline)*UPSAMPLE).astype(int)], 255, cv2.LINE_AA)

            # setting holes in building back to 0
            # all building elements but the first one are considered holes
            if len(building_elements) > 1:
                for j in range(1, len(building_elements)):
                    building_hole = building_elements[j]
                    first_coord = building_hole[0]
                    if len(first_coord) == 3:
                        building_hole = [coord[:2] for coord in building_hole]
                    cv2.fillPoly(mask, [np.rint(np.array(building_hole)).astype(int)], 0, cv2.LINE_AA)
                    cv2.fillPoly(mask_upsample, [np.rint(np.array(building_hole)*UPSAMPLE).astype(int)], 0, cv2.LINE_AA)

        # TODO: Maybe have a different threshold here for better building separation
        mask[mask<255] = 0
        mask_upsample[mask_upsample<255] = 0

        # saving created mask or show it
        if save:
            save_folder = aoi_folder / 'labels_raster'
            save_folder.mkdir(exist_ok=True)
            file = save_folder / f'{image_file.stem}_Buildings.tif'
            write_tif(file, mask, transform, crs)

            # Save upsampled images
            save_folder = aoi_folder / f'labels_raster_X{UPSAMPLE}'
            save_folder.mkdir(exist_ok=True)
            file = save_folder / f'{image_file.stem}_Buildings.tif'
            write_tif(file, mask_upsample, transform, crs)
        else:
            fig, axs = plt.subplots(1, 2, figsize=(10, 6))
            axs[0].imshow(img)
            axs[1].imshow(mask, interpolation='nearest')
            for ax in axs:
                ax.set_axis_off()
            plt.show()


# for normal split use pass ('train', 'validation') as split
def create_samples_file(dataset: str, split: tuple = None, split_ratio: float = 0.1):
    np.random.seed(SEED)
    # container for all the samples (each image results in a sample)
    samples = []
    # container to store all dates of a time series (aoi_id = dates)
    aoi_dates = {}

    aoi_ids = get_all_aoi_ids(dataset)
    random_numbers = list(np.random.rand(len(aoi_ids)))

    for aoi_id, random_number in zip(aoi_ids, random_numbers):
        all_dates_sorted = get_all_dates(dataset, aoi_id, sort_by_date=True)
        aoi_dates[aoi_id] = all_dates_sorted
        for i, (year, month) in enumerate(all_dates_sorted):
            sample = {'aoi_id': aoi_id, 'year': year, 'month': month, 'ts_index': i}
            if split is not None:
                sample['split'] = split[0] if random_number > split_ratio else split[1]
            samples.append(sample)

    file = DATASET_FOLDER / f'{dataset}_samples.json'
    save_json(file, {'samples': samples, 'aoi_dates': aoi_dates})


if __name__ == '__main__':

    # test_aoi = 'L15-1438E-1134N_5753_3655_13'
    # create_building_masks(test_aoi, save=False)

    # for aoi_id in sorted(get_all_aoi_ids()):
    #     create_building_masks(aoi_id, save=True)

    create_samples_file('train', split=('train', 'validation'), split_ratio=0.1)
    create_samples_file('test')
