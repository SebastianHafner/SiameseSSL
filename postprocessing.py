import cv2
import numpy as np
import shapely.geometry
import shapely.ops
import os
import sys
import math
import time
import torch
import concurrent.futures
from glob import glob
import pathlib
from tqdm import tqdm
from pathlib import Path
import geopandas
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from functools import partial

import utils
from train_network import inference_loop
from networks.network_loader import load_network
from experiment_manager.config import config
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



DATASET_PATH = Path('/storage/shafner/spacenet7')
OUTPUT_PATH = Path('/storage/shafner/spacenet7/outputs')
CONFIG_PATH = Path('/home/shafner/spacenet7/configs')
NETWORK_PATH = Path('/storage/shafner/spacenet7/networks/')


# input image should be uint8 [0, 255]
def find_contours(image: np.ndarray, wash: int = 64, kernel: int = 11, thresh: int = -1) -> tuple:
    # Threshold the image
    image[image < wash] = wash
    thresholded_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                              kernel, thresh)
    # Find contours
    # TODO: do something with holey buildings
    contours, hierarchy = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return thresholded_image, contours, hierarchy


# extracts all building polygons from a 2d array of probabilities [0, 1]
def segmentation2polygons(img: np.ndarray, cfg, min_building_size: float = 3.5,
                          simplification_tolerance: float = None) -> list:

    # rescaling and extracting building contours
    img_rescaled = np.squeeze(np.around(img * 255).astype(np.uint8))
    img_thresh, contours, hierarchy = find_contours(img_rescaled)

    # converting contours to shapely polygons and throwing away small buildings
    polygons = [Polygon(np.squeeze(c) / cfg.DATALOADER.UPSAMPLE) for c in contours if len(c) >= 3]
    polygons = [c for c in polygons if c.area > min_building_size]

    polygons = validate_polygons(polygons)

    if simplification_tolerance is not None:
        polygons = [c.simplify(tolerance=simplification_tolerance) for c in polygons]

    return polygons


def validate_polygons(polygons):
    # This messes up the shape somewhat for very small non-valid polygons
    valid_polys = []
    for p in polygons:
        red_poly = p.buffer(0)
        if isinstance(red_poly, shapely.geometry.MultiPolygon):
            if red_poly.geoms is not None:
                for g in red_poly.geoms:
                    if g is not None and g.is_valid and not g.is_empty:
                        valid_polys.append(g)
        elif red_poly is not None and red_poly.is_valid and not red_poly.is_empty:
            valid_polys.append(red_poly)
    return valid_polys


def inference_loop(net, cfg, device, callback=None, batch_size=1, max_samples=sys.maxsize,
                   dataset=None, callback_include_x=False, shuffle=True, drop_last=True, tta=False):
    if tta:
        transforms = tta.Compose(
            [
                tta.VerticalFlip(),
                tta.HorizontalFlip(),
                tta.Rotate90(angles=[0, 90, 180, 270]),
            ]
        )
        net = tta.SegmentationTTAWrapper(net, transforms, merge_mode='mean')

    net.to(device)
    net.eval()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=cfg.DATALOADER.NUM_WORKER,
                                       shuffle=shuffle, drop_last=drop_last)

    dataset_length = np.minimum(len(dataset), max_samples)
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):

            imgs = batch['x'].to(device)
            y_label = batch['y']
            aoi_id = batch['aoi_id']
            year = batch['year']
            month = batch['month']

            y_pred = net(imgs)
            y_pred = torch.sigmoid(y_pred)

            if callback:
                with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                    for i, yl, yp, ai, y, m in zip(imgs, y_label, y_pred, aoi_id, year, month):
                        executor.submit(callback, i, yl, yp, ai, y, m)

            if (max_samples is not None) and step >= max_samples:
                break


def process_images(config_name, checkpoint, run_type):
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')

    if run_type == 'test':
        dataset = utils.dataloader.SpaceNet7Dataset(cfg=cfg, dataset='test', no_augmentations=True)
    else:
        dataset = utils.dataloader.SpaceNet7Dataset(cfg=cfg, dataset='train', split=run_type, no_augmentations=True)

    # loading network
    checkpoint_path = NETWORK_PATH / f'{config_name}_{checkpoint}.pkl'
    print(f'Loading network {checkpoint_path}... ', end='', flush=True)
    net = load_network(cfg, checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    print('done!')

    def evaluate(img, y_label, y_pred, aoi_id, year, month):

        # setting up directory
        output_path = OUTPUT_PATH / config_name / run_type / aoi_id
        aoi_date = f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}'
        path_root = output_path / aoi_date
        output_path.mkdir(exist_ok=True)

        y_pred = y_pred.detach().cpu().numpy()
        polygons = segmentation2polygons(y_pred, cfg, min_building_size=3.5, simplification_tolerance=1)

        df = geopandas.GeoDataFrame(polygons, columns=['geometry'])
        df.to_file(f'{path_root}.geojson', driver="GeoJSON")

        # TODO: this is some extra stuff
        y_pred_cv2 = np.squeeze(np.around(y_pred * 255).astype(np.uint8))
        cv2.imwrite(f'{path_root}.png', y_pred_cv2)


    inference_loop(net, cfg, device, evaluate,
        batch_size=cfg.TRAINER.BATCH_SIZE,
        dataset=dataset,
        callback_include_x=False,
        shuffle=False,
        drop_last=False)


def iou(series, pred_poly_geom, iou_threshold, pred_poly_area):
    test_poly = series.geometry
    intersection = pred_poly_geom.intersection(test_poly).area
    # Save time by not calculating union all the time
    if intersection / float(pred_poly_area) < iou_threshold:
        return 0.0
    union = pred_poly_geom.union(test_poly).area
    # Calculate iou
    return intersection / float(union)


def match_proposals(pred_poly, test_data_GDF, iou_threshold, rtree, index_by_id):
    pred_poly_geom = pred_poly.geometry

    indexes = [index_by_id[id(pt)] for pt in rtree.query(pred_poly_geom)]
    rough_matches = test_data_GDF.loc[indexes]
    if len(rough_matches) == 0:
        return np.atleast_1d(pred_poly.fallback_id)

    precise_matches = rough_matches[rough_matches.intersects(pred_poly_geom)]
    if len(precise_matches) == 0:
        return np.atleast_1d(pred_poly.fallback_id)

    iou_partial = partial(iou, pred_poly_geom=pred_poly_geom, iou_threshold=iou_threshold,
                          pred_poly_area=pred_poly.area)
    iou_scores = precise_matches.apply(iou_partial, axis=1).to_numpy()
    indices = precise_matches.index.to_numpy()
    sorter = np.argsort(iou_scores)
    cutoff = np.searchsorted(iou_scores, iou_threshold, side='right', sorter=sorter)

    return np.append(np.take_along_axis(indices, sorter[cutoff:][::-1], axis=0), pred_poly.fallback_id)


# TODO: This misses a few matches tha the original one finds, not sure which one is buggy
def track_footprint_identifiers(filenames,
                                min_iou=0.25,
                                reverse_order=False,
                                verbose=True, super_verbose=False):
    # set columns for master gdf
    gdf_master_columns = ['id', 'area', 'geometry']

    gdf_list = []
    for j, f in enumerate(tqdm(filenames)):
        gdf_now = geopandas.read_file(f)

        gdf_now['area'] = gdf_now['geometry'].area
        gdf_now['id'] = -1

        # sort by reverse area
        gdf_now.sort_values(by=['area'], ascending=False, inplace=True)
        gdf_now = gdf_now.reset_index(drop=True)

        # reorder columns (if needed)
        gdf_now = gdf_now[gdf_master_columns]

        if len(gdf_now) == 1 and gdf_now.geometry[0].is_empty:
            gdf_now.to_csv(str(f) + '.csv', index=False)
            gdf_list.append(gdf_now)
            continue

        if verbose:
            print("\n")
            print("", j, "file_name:", f.stem)
            print("  ", "gdf_now.columns:", gdf_now.columns)

        if j == 0:
            # Establish initial footprints at Epoch0
            gdf_now['id'] = gdf_now.index.values
            gdf_now['fallback_id'] = -1
            gdf_now['prop_ids'] = gdf_now.index.values
            new_id = len(gdf_now)
            gdf_master = gdf_now.copy(deep=True)

        else:

            if verbose:
                print("   len gdf_now:", len(gdf_now), "len(gdf_master):", len(gdf_master),
                      "max master id:", np.max(gdf_master['id']))

            new_id = np.max(gdf_master['id']) + 1

            gdf_now['fallback_id'] = np.arange(new_id, new_id + len(gdf_now))

            rtree = STRtree(gdf_master.geometry)
            index_by_id = {id(pt): i for i, pt in zip(gdf_master.index, gdf_master.geometry)}
            match_proposals_patrial = partial(match_proposals, test_data_GDF=gdf_master, iou_threshold=min_iou,
                                              rtree=rtree, index_by_id=index_by_id)
            gdf_now['prop_ids'] = gdf_now.apply(match_proposals_patrial, axis=1)
            gdf_now['id'] = gdf_now['prop_ids'].apply(lambda x: x[0])

            # Iterate proposals until no duplicates remain
            mask = gdf_now.duplicated('id', keep='first')
            index = 1
            while mask.any():
                gdf_now.loc[mask, 'id'] = gdf_now.loc[mask, 'prop_ids'].apply(lambda x: x[index])
                # logical and so that once a row is not duplicated it never will be again
                mask = np.logical_and(mask, gdf_now.duplicated('id', keep='first'))
                index += 1

            # Squish
            new_id_mask = gdf_now['id'] == gdf_now['fallback_id']
            gdf_now.loc[new_id_mask, 'id'] = np.arange(new_id, new_id + np.sum(new_id_mask))

            # Add any new id-s to master
            gdf_master = gdf_master.append(gdf_now[new_id_mask])

        gdf_now.to_csv(str(f) + '.csv', index=False)
        gdf_list.append(gdf_now)

    return gdf_list


def process_polygons(config_name, run_type):
    dir = OUTPUT_PATH / config_name / run_type
    aoi_id_dirs = sorted(dir.glob('*'))

    gdf_list = []
    gdf_files_list = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for aoi_id_dir in aoi_id_dirs:
            print(f'Processing directory {aoi_id_dir}')
            aoi_date_files = sorted(aoi_id_dir.glob('*.geojson'))[::-1]
            # gdf_files = [geopandas.read_file(f) for f in aoi_date_files]
            gdf_files_list.extend(aoi_date_files)
            futures.append(executor.submit(track_footprint_identifiers, aoi_date_files))
        for f in futures:
            gdf_list.extend(f.result())

    net_df = None
    for df, fname in zip(gdf_list, gdf_files_list):
        df = geopandas.GeoDataFrame({
            'filename': fname.stem,
            'id': df.id.astype(int),
            'geometry': df.geometry,
        })
        if len(df) == 0:
            message = '! Empty dataframe for %s' % json_file
            print(message)
            # raise Exception(message)
        if net_df is None:
            net_df = df
        else:
            net_df = net_df.append(df)

    output_csv_path = OUTPUT_PATH / config_name / run_type / 'proposals.csv'
    net_df.to_csv(output_csv_path, index=False)


if __name__ == '__main__':
    config_name = 'threeimages'
    checkpoint = 100
    run_type = 'test'

    # process_images(config_name, checkpoint, run_type)
    start_time = time.time()
    process_polygons(config_name, run_type)
    execution_time = time.time() - start_time
    print(execution_time)
