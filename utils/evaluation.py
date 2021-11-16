import torch
from torch.utils import data as torch_data
import wandb
from tqdm import tqdm
from utils import datasets, metrics


def model_evaluation(net, cfg, device, run_type: str, epoch: float, step: int, enable_sem: bool = False):
    thresholds = torch.linspace(0.5, 1, 1).to(device)
    measurer_change = metrics.MultiThresholdMetric(thresholds)
    measurer_sem = metrics.MultiThresholdMetric(thresholds)
    ds = datasets.SpaceNet7CDDataset(cfg, run_type, no_augmentations=True, dataset_mode='first_last',
                                     disable_multiplier=True)

    net.to(device)
    net.eval()

    # reset the generators
    num_workers = 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER
    dataloader = torch_data.DataLoader(ds, batch_size=1, num_workers=num_workers, shuffle=False, drop_last=False)

    with torch.no_grad():
        for step, item in enumerate(dataloader):

            x_t1 = item['x_t1'].to(device)
            x_t2 = item['x_t2'].to(device)
            logits_change, logits_sem_t1, logits_sem_t2 = net(x_t1, x_t2)
            y_pred_change = torch.sigmoid(logits_change)

            gt_change = item['y_change'].to(device)
            measurer_change.add_sample(gt_change.detach(), y_pred_change.detach())

            if enable_sem:
                y_pred_sem_t1 = torch.sigmoid(logits_sem_t1)
                gt_sem_t1 = item['y_sem_t1'].to(device)
                measurer_sem.add_sample(gt_sem_t1, y_pred_sem_t1)

                y_pred_sem_t2 = torch.sigmoid(logits_sem_t2)
                gt_sem_t2 = item['y_sem_t2'].to(device)
                measurer_sem.add_sample(gt_sem_t2, y_pred_sem_t2)

            if cfg.DEBUG:
                break

    print(f'Computing {run_type} F1 score ', end=' ', flush=True)

    f1s_change = measurer_change.compute_f1()
    precisions_change, recalls_change = measurer_change.precision, measurer_change.recall

    # best f1 score for passed thresholds
    f1_change = f1s_change.max()
    argmax_f1_change = f1s_change.argmax()

    precision_change = precisions_change[argmax_f1_change]
    recall_change = recalls_change[argmax_f1_change]

    print(f'{f1_change.item():.3f}', flush=True)

    if not cfg.DEBUG:
        wandb.log({f'{run_type} change F1': f1_change,
                   f'{run_type} change precision': precision_change,
                   f'{run_type} change recall': recall_change,
                   'step': step, 'epoch': epoch,
                   })
        if enable_sem:
            f1s_sem = measurer_sem.compute_f1()
            precisions_sem, recalls_sem = measurer_sem.precision, measurer_sem.recall

            # best f1 score for passed thresholds
            f1_sem = f1s_sem.max()
            argmax_f1_sem = f1s_sem.argmax()

            precision_sem = precisions_sem[argmax_f1_sem]
            recall_sem = recalls_sem[argmax_f1_sem]
            wandb.log({f'{run_type} sem F1': f1_sem,
                       f'{run_type} sem precision': precision_sem,
                       f'{run_type} sem recall': recall_sem,
                       'step': step, 'epoch': epoch,
                       })