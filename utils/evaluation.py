import torch
from torch.utils import data as torch_data
import wandb
from tqdm import tqdm
from utils import datasets, metrics


def model_evaluation(net, cfg, device, run_type: str, epoch: float, step: int):
    thresholds = torch.linspace(0.5, 1, 1).to(device)
    measurer = metrics.MultiThresholdMetric(thresholds)
    ds = datasets.SpaceNet7CDDataset(cfg, run_type, no_augmentations=True, dataset_mode='first_last',
                                     disable_multiplier=True)

    net.to(device)
    net.eval()

    # reset the generators
    num_workers = 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER
    dataloader = torch_data.DataLoader(ds, batch_size=1, num_workers=num_workers, shuffle=False, drop_last=False)

    with torch.no_grad():
        for step, item in enumerate(tqdm(dataloader)):

            x_t1 = item['x_t1'].to(device)
            x_t2 = item['x_t2'].to(device)
            y_pred = net(x_t1, x_t2)
            y_pred = torch.sigmoid(y_pred)

            y_gts = item['y'].to(device)
            measurer.add_sample(y_gts.detach(), y_pred.detach())

            if cfg.DEBUG:
                break

    print(f'Computing {run_type} F1 score ', end=' ', flush=True)

    f1s = measurer.compute_f1()
    precisions, recalls = measurer.precision, measurer.recall

    # best f1 score for passed thresholds
    f1 = f1s.max()
    argmax_f1 = f1s.argmax()

    precision = precisions[argmax_f1]
    recall = recalls[argmax_f1]

    print(f'{f1.item():.3f}', flush=True)

    if not cfg.DEBUG:
        wandb.log({f'{run_type} change F1': f1,
                   f'{run_type} change precision': precision,
                   f'{run_type} change recall': recall,
                   'step': step, 'epoch': epoch,
                   })
