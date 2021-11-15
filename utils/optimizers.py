import torch.optim

def optimizer_from_cfg(cfg, params):
    if cfg.TRAINER.ALGORITHM == 'adam':
        return torch.optim.Adam(params, lr=cfg.TRAINER.LR, weight_decay=cfg.TRAINER.WEIGHT_DECAY)
    if cfg.TRAINER.ALGORITHM == 'adamw':
        return torch.optim.AdamW(params, lr=cfg.TRAINER.LR, weight_decay=cfg.TRAINER.WEIGHT_DECAY)
    if cfg.TRAINER.ALGORITHM == 'sgd':
        return torch.optim.SGD(params, lr=cfg.TRAINER.LR, weight_decay=cfg.TRAINER.WEIGHT_DECAY, nesterov=True)
    else:
        raise Exception(f'Unknown optimizer algorithm {cfg.TRAINER.ALGORITHM}')
