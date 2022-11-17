import sys
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

from tabulate import tabulate
import wandb
import numpy as np
from pathlib import Path

from utils import networks, datasets, loss_functions, evaluation, experiment_manager


def run_training(cfg):
    run_config = {
        'CONFIG_NAME': cfg.NAME,
        'device': device,
        'epochs': cfg.TRAINER.EPOCHS,
        'learning rate': cfg.TRAINER.LR,
        'batch size': cfg.TRAINER.BATCH_SIZE,
    }
    table = {'run config name': run_config.keys(),
             ' ': run_config.values(),
             }
    print(tabulate(table, headers='keys', tablefmt="fancy_grid", ))

    net = networks.create_network(cfg)
    net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)

    # reset the generators
    dataset = datasets.SpaceNet7CDDataset(cfg=cfg, run_type='training')
    print(dataset)

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

    # early stopping
    best_f1_change_val = 0
    trigger_times = 0
    stop_training = False

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        loss_set = []

        for i, batch in enumerate(dataloader):

            net.train()
            optimizer.zero_grad()

            x_t1 = batch['x_t1'].to(device)
            x_t2 = batch['x_t2'].to(device)

            logits, _, _ = net(x_t1, x_t2)

            gt_change = batch['y_change'].to(device)

            loss = criterion(logits, gt_change)
            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOGGING.FREQUENCY == 0:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')
                # evaluation on sample of training and validation set
                _ = evaluation.model_evaluation(net, cfg, device, 'training', epoch_float, global_step)
                _ = evaluation.model_evaluation(net, cfg, device, 'validation', epoch_float, global_step)

                # logging
                time = timeit.default_timer() - start
                wandb.log({
                    'loss': np.mean(loss_set),
                    'labeled_percentage': 100,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_set = []
            # end of batch

        assert (epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        # evaluation at the end of an epoch
        _ = evaluation.model_evaluation(net, cfg, device, 'training', epoch_float, global_step)
        f1_change_val = evaluation.model_evaluation(net, cfg, device, 'validation', epoch_float, global_step)
        _ = evaluation.model_evaluation(net, cfg, device, 'test', epoch_float, global_step)

        if cfg.EARLY_STOPPING.ENABLE:
            if f1_change_val <= best_f1_change_val:
                trigger_times += 1
                if trigger_times > cfg.SAVE.PATIENCE:
                    stop_training = True
            else:
                best_f1_change_val = f1_change_val
                print(f'saving network (F1 {f1_change_val:.3f})', flush=True)
                networks.save_checkpoint(net, optimizer, epoch, global_step, cfg, early_stopping=True)
                trigger_times = 0

        if epoch == cfg.TRAINER.EPOCHS and not cfg.DEBUG:
            print(f'saving network (end of training)', flush=True)
            networks.save_checkpoint(net, optimizer, epoch, global_step, cfg)

        if stop_training:
            break  # end of training by early stopping

    # final logging for early stopping
    if cfg.EARLY_STOPPING.ENABLE:
        net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device, best_val=True)
        evaluation.model_evaluation_earlystopping(net, cfg, device, 'training')
        evaluation.model_evaluation_earlystopping(net, cfg, device, 'validation')
        evaluation.model_evaluation_earlystopping(net, cfg, device, 'test')


if __name__ == '__main__':
    args = experiment_manager.default_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    # make training deterministic
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('=== Runnning on device: p', device)

    wandb.init(
        name=cfg.NAME,
        config=cfg,
        project='siamese_ssl_extended',
        entity='spacenet7',
        tags=['ssl', 'cd', 'siamese', 'spacenet7', ],
        mode='online' if not cfg.DEBUG else 'disabled',
    )

    try:
        run_training(cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
