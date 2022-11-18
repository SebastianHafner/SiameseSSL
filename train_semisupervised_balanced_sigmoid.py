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


def concatenate_batches(batch1, batch2, dict_key: str):
    return torch.cat((batch1[dict_key], batch2[dict_key]), dim=0)


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
    change_consistency_criterion = loss_functions.get_criterion(cfg.CONSISTENCY_TRAINER.LOSS_TYPE)

    # reset the generators
    dataset_labeled = datasets.SpaceNet7CDDataset(cfg=cfg, run_type='training', disable_unlabeled=True)
    dataset_unlabeled = datasets.SpaceNet7CDDataset(cfg=cfg, run_type='training', only_unlabeled=True)

    total_samples = len(dataset_labeled) + len(dataset_unlabeled)
    fraction_labeled = len(dataset_labeled) / total_samples
    fraction_unlabeled = len(dataset_unlabeled) / total_samples
    batch_size_labeled = int(np.rint(cfg.TRAINER.BATCH_SIZE * fraction_labeled))
    batch_size_unlabeled = int(np.rint(cfg.TRAINER.BATCH_SIZE * fraction_unlabeled))
    assert (batch_size_labeled + batch_size_unlabeled == cfg.TRAINER.BATCH_SIZE)

    dataloader_kwargs = {
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }

    dataloader_labeled = torch_data.DataLoader(dataset_labeled, batch_size=batch_size_labeled, **dataloader_kwargs)
    dataloader_unlabeled = torch_data.DataLoader(dataset_unlabeled, batch_size=batch_size_unlabeled,
                                                 **dataloader_kwargs)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    steps_per_epoch = len(dataloader_labeled) if len(dataloader_labeled) < len(dataloader_unlabeled) else \
        len(dataloader_unlabeled)

    # tracking variables
    global_step = epoch_float = 0

    # early stopping
    best_f1_change_val = 0
    trigger_times = 0
    stop_training = False

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        loss_set, sem_loss_set, change_loss_set, change_sem_loss_set, consistency_loss_set = [], [], [], [], []
        n_labeled, n_notlabeled = 0, 0

        for i, (batch_labeled, batch_unlabeled) in enumerate(zip(dataloader_labeled, dataloader_unlabeled)):

            if i == steps_per_epoch:
                break

            net.train()
            optimizer.zero_grad()

            x_t1 = concatenate_batches(batch_labeled, batch_unlabeled, 'x_t1').to(device)
            x_t2 = concatenate_batches(batch_labeled, batch_unlabeled, 'x_t2').to(device)

            logits_change, logits_sem_t1, logits_sem_t2 = net(x_t1, x_t2)
            if cfg.MODEL.ENABLE_SEMANTIC_CHANGE_OUTCONV:
                logits_change_sem = net.module.outc_sem_change(torch.cat((logits_sem_t1, logits_sem_t2), dim=1))
                y_pred_change_sem = torch.sigmoid(logits_change_sem)
            else:
                y_pred_change_sem = torch.sub(torch.sigmoid(logits_sem_t2), torch.sigmoid(logits_sem_t1))

            supervised_loss, consistency_loss = None, None

            is_labeled = concatenate_batches(batch_labeled, batch_unlabeled, 'is_labeled')
            n_labeled += torch.sum(is_labeled).item()
            if is_labeled.any():
                # change detection
                gt_change = concatenate_batches(batch_labeled, batch_unlabeled, 'y_change').to(device)
                change_loss = change_criterion(logits_change[is_labeled,], gt_change[is_labeled,])

                # semantic segmentation
                gt_sem_t1 = concatenate_batches(batch_labeled, batch_unlabeled, 'y_sem_t1').to(device)
                gt_sem_t2 = concatenate_batches(batch_labeled, batch_unlabeled, 'y_sem_t2').to(device)

                sem_t1_loss = sem_criterion(logits_sem_t1[is_labeled,], gt_sem_t1[is_labeled,])
                sem_t2_loss = sem_criterion(logits_sem_t2[is_labeled,], gt_sem_t2[is_labeled,])

                sem_loss = (sem_t1_loss + sem_t2_loss) / 2
                supervised_loss = change_loss + sem_loss

                if cfg.MODEL.ENABLE_SEMANTIC_CHANGE_LOSS:
                    sem_change_loss = change_criterion(y_pred_change_sem[is_labeled,], gt_change[is_labeled,],
                                                       disable_sigmoid=True)
                    change_sem_loss_set.append(sem_change_loss.item())
                    supervised_loss = supervised_loss + sem_change_loss

                sem_loss_set.append(sem_loss.item())
                change_loss_set.append(change_loss.item())

            if not is_labeled.all():
                is_not_labeled = torch.logical_not(is_labeled)
                n_notlabeled += torch.sum(is_not_labeled).item()

                if cfg.CONSISTENCY_TRAINER.LOSS_TYPE == 'L2':
                    y_pred_change = torch.sigmoid(logits_change)
                    consistency_loss = change_consistency_criterion(y_pred_change[is_not_labeled,],
                                                                    y_pred_change_sem[is_not_labeled,])
                else:
                    consistency_loss = change_consistency_criterion(logits_change[is_not_labeled,],
                                                                    y_pred_change_sem[is_not_labeled,])
                consistency_loss = consistency_loss * cfg.CONSISTENCY_TRAINER.LOSS_FACTOR
                consistency_loss_set.append(consistency_loss.item())

            if supervised_loss is None and consistency_loss is not None:
                loss = consistency_loss
            elif supervised_loss is not None and consistency_loss is not None:
                loss = supervised_loss + consistency_loss
            else:
                loss = supervised_loss

            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOGGING.FREQUENCY == 0:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')

                # evaluation on sample of training and validation set
                _ = evaluation.model_evaluation(net, cfg, device, 'training', epoch_float, global_step, enable_sem=True)
                _ = evaluation.model_evaluation(net, cfg, device, 'validation', epoch_float, global_step, enable_sem=True)

                # logging
                time = timeit.default_timer() - start
                wandb.log({
                    'change_loss': np.mean(change_loss_set) if len(change_loss_set) > 0 else 0,
                    'change_sem_loss': np.mean(change_sem_loss_set) if len(change_sem_loss_set) > 0 else 0,
                    'sem_loss': np.mean(sem_loss_set) if len(sem_loss_set) > 0 else 0,
                    'cons_loss': np.mean(consistency_loss_set) if len(consistency_loss_set) > 0 else 0,
                    'loss': np.mean(loss_set),
                    'labeled_percentage': n_labeled / (n_labeled + n_notlabeled) * 100,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                n_labeled, n_notlabeled = 0, 0
                loss_set, sem_loss_set, change_loss_set, change_sem_loss_set, consistency_loss_set = [], [], [], [], []
            # end of batch

        assert (epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        # evaluation at the end of an epoch
        _ = evaluation.model_evaluation(net, cfg, device, 'training', epoch_float, global_step, enable_sem=True)
        f1_change_val = evaluation.model_evaluation(net, cfg, device, 'validation', epoch_float, global_step,
                                                    enable_sem=True)
        _ = evaluation.model_evaluation(net, cfg, device, 'test', epoch_float, global_step, enable_sem=True)

        if cfg.EARLY_STOPPING.ENABLE:
            if f1_change_val <= best_f1_change_val:
                trigger_times += 1
                if trigger_times > cfg.EARLY_STOPPING.PATIENCE:
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
        net, *_ = networks.load_checkpoint(cfg.TRAINER.EPOCHS, cfg, device, best_val=True)
        evaluation.model_evaluation_earlystopping(net, cfg, device, 'training', enable_sem=True)
        evaluation.model_evaluation_earlystopping(net, cfg, device, 'validation', enable_sem=True)
        evaluation.model_evaluation_earlystopping(net, cfg, device, 'test', enable_sem=True)


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
