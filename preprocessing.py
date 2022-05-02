import argparse
from pathlib import Path
import numpy as np
from utils import geofiles, spacenet7_helpers
import matplotlib.pyplot as plt


def round_to_255(img):
    img[img > 127] = 255
    img[img <= 127] = 0
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
        mask_upsample = np.zeros((img.shape[0] * UPSAMPLE, img.shape[1] * UPSAMPLE, 1), dtype=np.uint8)

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
            cv2.fillPoly(mask_upsample, [np.rint(np.array(building_outline) * UPSAMPLE).astype(int)], 255, cv2.LINE_AA)

            # setting holes in building back to 0
            # all building elements but the first one are considered holes
            if len(building_elements) > 1:
                for j in range(1, len(building_elements)):
                    building_hole = building_elements[j]
                    first_coord = building_hole[0]
                    if len(first_coord) == 3:
                        building_hole = [coord[:2] for coord in building_hole]
                    cv2.fillPoly(mask, [np.rint(np.array(building_hole)).astype(int)], 0, cv2.LINE_AA)
                    cv2.fillPoly(mask_upsample, [np.rint(np.array(building_hole) * UPSAMPLE).astype(int)], 0,
                                 cv2.LINE_AA)

        # TODO: Maybe have a different threshold here for better building separation
        mask[mask < 255] = 0
        mask_upsample[mask_upsample < 255] = 0

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


def create_metadata_file(spacenet7_path: str, dataset: str):

    # container to store all dates of a time series (aoi_id = dates)
    metadata = {}

    for aoi_id in spacenet7_helpers.get_all_aoi_ids(spacenet7_path, dataset):
        # container for all the timestamps (each image results in a timestamp)
        timestamps = []
        all_dates_sorted = spacenet7_helpers.get_all_dates(spacenet7_path, aoi_id, sort_by_date=True)
        for i, date in enumerate(all_dates_sorted):
            year, month = date
            timestamp = {
                'aoi_id': aoi_id,
                'index': i,
                'year': year,
                'month': month,
                'mask': spacenet7_helpers.is_masked(spacenet7_path, aoi_id, year, month),
                'label': True if dataset == 'train' else False,
            }
            timestamps.append(timestamp)

        metadata[aoi_id] = timestamps

    file = Path(spacenet7_path) / f'metadata_{dataset}.json'
    geofiles.write_json(file, metadata)


def dataset_split(spacenet7_path: str, dataset: str, seed: int = 42):
    aoi_ids = spacenet7_helpers.get_all_aoi_ids(spacenet7_path, dataset)
    np.random.seed(seed)
    rand_numbers = np.random.rand(len(aoi_ids))
    splits = [[], [], []]
    for aoi_id, rand_number in zip(aoi_ids, rand_numbers):
        if rand_number < 0.6:
            splits[0].append(aoi_id)
        elif rand_number < 0.8:
            splits[1].append(aoi_id)
        else:
            splits[2].append(aoi_id)
    for split, aoi_ids in zip(['training', 'validation', 'test'], splits):
        print(split)
        for aoi_id in aoi_ids:
            print(f"'{aoi_id}',")


def metadata_argument_parser():
    # https://docs.python.org/3/library/argparse.html#the-add-argument-method
    parser = argparse.ArgumentParser(description="Experiment Args")

    parser.add_argument('-s', "--spacenet7-dir", dest='spacenet7_dir', required=True, help="path to SpaceNet7 dataset")
    parser.add_argument('-d', "--dataset", dest='dataset', required=True, help="dataset (train/test)")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    args = metadata_argument_parser().parse_known_args()[0]
    create_metadata_file(args.spacenet7_dir, args.dataset)
    # dataset_split(args.spacenet7_dir, args.dataset)