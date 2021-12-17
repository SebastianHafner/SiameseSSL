import ee
import argparse
from pathlib import Path
from utils import experiment_manager, spacenet7_helpers, geofiles


def get_centroid(aoi_id: str, spacenet7_path: str, dataset: str) -> ee.Geometry:
    folder = Path(spacenet7_path) / dataset / aoi_id / 'images_masked'
    files = [file for file in folder.glob('**/*') if file.is_file()]
    _, transform, crs = geofiles.read_tif(files[0])
    _, _, c, _, _, f, *_ = transform
    return ee.Geometry.Point(coords=[c, f], proj=str(crs)).transform()


def upload_study_area(spacenet7_path: str):
    cfg = experiment_manager.load_cfg('base')
    train_aoi_ids = spacenet7_helpers.get_all_aoi_ids(spacenet7_path, 'train')
    test_aoi_ids = spacenet7_helpers.get_all_aoi_ids(spacenet7_path, 'test')
    features = []
    for aoi_id in train_aoi_ids + test_aoi_ids:
        centroid = get_centroid(aoi_id, spacenet7_path, 'train' if aoi_id in train_aoi_ids else 'test')
        if aoi_id in cfg.DATASET.TRAINING_IDS:
            split = 'training'
            labeled = 1
        elif aoi_id in cfg.DATASET.VALIDATION_IDS:
            split = 'validation'
            labeled = 1
        elif aoi_id in cfg.DATASET.TEST_IDS:
            split = 'test'
            labeled = 1
        else:
            split = 'training'
            labeled = 0
        features.append(ee.Feature(centroid, {'aoi_id': aoi_id, 'split': split, 'labeled': labeled}))

    fc = ee.FeatureCollection(features)
    dl_task = ee.batch.Export.table.toDrive(
        collection=fc,
        description='siameseSSLstudyArea',
        folder='siamese_ssl',
        fileNamePrefix='siamese_ssl_aoi_ids',
        fileFormat='GeoJSON'
    )
    dl_task.start()


def metadata_argument_parser():
    # https://docs.python.org/3/library/argparse.html#the-add-argument-method
    parser = argparse.ArgumentParser(description="Experiment Args")

    parser.add_argument('-s', "--spacenet7-dir", dest='spacenet7_dir', required=True, help="path to SpaceNet7 dataset")
    parser.add_argument('-c', "--config-file", dest='config_file', required=False, default="base",
                        help="path to config file")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    ee.Initialize()
    args = metadata_argument_parser().parse_known_args()[0]
    upload_study_area(args.spacenet7_dir)