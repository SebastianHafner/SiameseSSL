# utility functions to extract information from configs


def n_sentinel1_bands(cfg) -> int:
    return len(cfg.DATALOADER.SENTINEL1_BANDS)


def n_sentinel2_bands(cfg) -> int:
    return len(cfg.DATALOADER.SENTINEL2_BANDS)


def n_bands(cfg):
    return n_sentinel1_bands(cfg) + n_sentinel2_bands(cfg)

