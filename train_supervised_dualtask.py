import sys
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

from tabulate import tabulate
import wandb
import numpy as np

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

    change_criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)
    sem_criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)

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
    save_checkpoints = cfg.SAVE_CHECKPOINTS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()

        loss_set, sem_loss_set, change_loss_set = [], [], []

        for i, batch in enumerate(dataloader):

            net.train()
            optimizer.zero_grad()

            x_t1 = batch['x_t1'].to(device)
            x_t2 = batch['x_t2'].to(device)
            logits_change, logits_sem_t1, logits_sem_t2 = net(x_t1, x_t2)

            # change detection
            gt_change = batch['y_change'].to(device)
            change_loss = change_criterion(logits_change, gt_change)

            # semantic segmentation
            gt_sem_t1 = batch['y_sem_t1'].to(device)
            gt_sem_t2 = batch['y_sem_t2'].to(device)

            sem_t1_loss = sem_criterion(logits_sem_t1, gt_sem_t1)
            sem_t2_loss = sem_criterion(logits_sem_t2, gt_sem_t2)
            sem_loss = (sem_t1_loss + sem_t2_loss) / 2

            loss = (change_loss + sem_loss) / 2

            loss.backward()
            optimizer.step()

            sem_loss_set.append(sem_loss.item())
            change_loss_set.append(change_loss.item())
            loss_set.append(loss.item())

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')

                # evaluation on sample of training and validation set
                evaluation.model_evaluation(net, cfg, device, 'training', epoch_float, global_step, enable_sem=True)
                evaluation.model_evaluation(net, cfg, device, 'validation', epoch_float, global_step, enable_sem=True)

                # logging
                time = timeit.default_timer() - start
                wandb.log({
                    'change_loss': np.mean(change_loss_set),
                    'sem_loss': np.mean(sem_loss_set),
                    'loss': np.mean(loss_set),
                    'labeled_percentage': 100,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_set, sem_loss_set, change_loss_set = [], [], []
            # end of batch

        assert (epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        # evaluation at the end of an epoch
        evaluation.model_evaluation(net, cfg, device, 'training', epoch_float, global_step, enable_sem=True)
        evaluation.model_evaluation(net, cfg, device, 'validation', epoch_float, global_step, enable_sem=True)
        evaluation.model_evaluation(net, cfg, device, 'test', epoch_float, global_step, enable_sem=True)

        if epoch in save_checkpoints and not cfg.DEBUG:
            print(f'saving network', flush=True)
            networks.save_checkpoint(net, optimizer, epoch, global_step, cfg)


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
        project='siamese_ssl',
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
