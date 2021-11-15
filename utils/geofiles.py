import rasterio
import json
from pathlib import Path
from shapely import wkt


# reading in geotiff file as numpy array
def read_tif(file: Path):

    if not file.exists():
        raise FileNotFoundError(f'File {file} not found')

    with rasterio.open(file) as dataset:
        arr = dataset.read()  # (bands X height X width)
        transform = dataset.transform
        crs = dataset.crs

    return arr.transpose((1, 2, 0)), transform, crs


# writing an array to a geo tiff file
def write_tif(file: Path, arr, transform, crs):

    if not file.parent.exists():
        file.parent.mkdir()

    height, width, bands = arr.shape
    with rasterio.open(
            file,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=bands,
            dtype=arr.dtype,
            crs=crs,
            transform=transform,
    ) as dst:
        for i in range(bands):
            dst.write(arr[:, :, i], i + 1)


def load_json(file: Path):
    with open(str(file)) as f:
        d = json.load(f)
    return d


def save_json(file: Path, content):
    with open(str(file), 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=4)


def load_polygons(filename):
    with open(filename) as f:
        return [wkt.loads(l) for l in f]


def save_polygons(filename, polygons):
    with open(filename, mode='w') as f:
        for p in polygons:
            f.write(f'{wkt.dumps(p)}\n')
