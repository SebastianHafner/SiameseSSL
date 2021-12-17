import torch
from torch.utils import data as torch_data
import wandb
from utils import datasets, metrics


def inference_loop(net, dataset: datasets.SpaceNet7CDDataset, device: str, enable_sem: bool = False) -> dict:

    net.to(device)
    net.eval()

    thresholds = torch.linspace(0.5, 1, 1).to(device)
    measurer_change = metrics.MultiThresholdMetric(thresholds)
    measurer_sem = metrics.MultiThresholdMetric(thresholds)

    data = {'change': [], 'semantics': []}

    dataloader = torch_data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False)
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

    f1s_change = measurer_change.compute_f1()
    precisions_change, recalls_change = measurer_change.precision, measurer_change.recall
    data['change'].append(f1s_change.max().item())
    argmax_f1_change = f1s_change.argmax()
    data['change'].append(precisions_change[argmax_f1_change].item())
    data['change'].append(recalls_change[argmax_f1_change].item())

    if enable_sem:
        f1s_sem = measurer_sem.compute_f1()
        precisions_sem, recalls_sem = measurer_sem.precision, measurer_sem.recall
        data['semantics'].append(f1s_sem.max().item())
        argmax_f1_sem = f1s_sem.argmax()
        data['semantics'].append(precisions_sem[argmax_f1_sem].item())
        data['semantics'].append(recalls_sem[argmax_f1_sem].item())

    return data


def model_evaluation(net, cfg, device, run_type: str, epoch: float, step: int, enable_sem: bool = False):

    ds = datasets.SpaceNet7CDDataset(cfg, run_type, no_augmentations=True, dataset_mode='first_last',
                                     disable_multiplier=True, disable_unlabeled=True)

    data = inference_loop(net, ds, device, enable_sem)

    f1_change, precision_change, recall_change = data['change']
    wandb.log({f'{run_type} change F1': f1_change,
               f'{run_type} change precision': precision_change,
           f'{run_type} change recall': recall_change,
               'step': step, 'epoch': epoch,
               })

    if enable_sem:
        f1_sem, precision_sem, recall_sem = data['semantics']
        wandb.log({f'{run_type} sem F1': f1_sem,
                   f'{run_type} sem precision': precision_sem,
                   f'{run_type} sem recall': recall_sem,
                   'step': step, 'epoch': epoch,
                   })