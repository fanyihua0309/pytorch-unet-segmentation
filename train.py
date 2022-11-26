import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from model import UNet
from dataset import SegmentationDataset
from loss import dice_loss
from utils import log_config


def train():
    wandb.init(project='unet_for_segmentation', resume='allow')
    logging = log_config('log/train.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    lr = 1e-4
    total_epoch = 30
    batch_size = 4

    wandb.config = {
        'learning_rate': lr,
        'epochs': total_epoch,
        'batch_size': batch_size
    }

    dataset = SegmentationDataset()
    logging.info(f'Train: {len(dataset)}')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    checkpoint_root = 'checkpoint'
    if not os.path.exists(checkpoint_root):
        os.mkdir(checkpoint_root)

    weight = 0.5
    min_loss = float('inf')
    model.train()
    for epoch in range(1, total_epoch + 1):
        step = 0
        interval_loss = 0
        with tqdm(total=len(loader), desc=f'Train epoch{epoch}') as pbar:
            for imgs, masks in loader:
                imgs = imgs.to(device)
                masks = masks.to(device, dtype=torch.long)
                preds = model(imgs)
                cross_entropy = criterion(preds, masks)
                dice = dice_loss(preds.max(dim=1)[1], masks)
                loss = weight * cross_entropy + (1 - weight) * dice

                step += 1
                interval_loss += loss.item()
                wandb.log({'loss': loss.item(), 'epoch': epoch})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(**{
                    'loss': f'{interval_loss / step: .6f}'
                })
                pbar.update(1)
            mean_loss = interval_loss / step
            wandb.log({'mean_loss': mean_loss, 'epoch': epoch})
            save_path = f'{checkpoint_root}/epoch{epoch}_loss{mean_loss: .4f}.pth'
            torch.save(model.state_dict(), save_path)
            if mean_loss < min_loss:
                min_loss = mean_loss
                save_path = f'{checkpoint_root}/best_model_epoch{epoch}_loss{mean_loss:.4f}.pth'
                torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    train()

