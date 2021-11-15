import torch

import segmentation_models_pytorch as smp
from networks.unet import UNet

from pathlib import Path


def create_network(cfg):

    architecture = cfg.MODEL.TYPE

    if architecture == 'unet':

        if cfg.MODEL.BACKBONE.ENABLED:
            net = smp.Unet(
                cfg.MODEL.BACKBONE.TYPE,
                encoder_weights=cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS,
                in_channels=cfg.MODEL.IN_CHANNELS,
                classes=cfg.MODEL.OUT_CHANNELS,
                activation=None,
            )
        else:
            net = UNet(cfg)

    else:
        net = UNet(cfg)

    return net


def load_network(cfg, pkl_file: Path):

    net = create_network(cfg)
    state_dict = torch.load(str(pkl_file), map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict)

    return net
