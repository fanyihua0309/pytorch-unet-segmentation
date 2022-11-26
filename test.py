import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from model import UNet
from dataset import SegmentationDataset
from metrics import StreamSegMetrics
import matplotlib.colors as colors
from utils import log_config


def test():
    wandb.init(project='unet_for_segmentation')
    logging = log_config('log/test.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    batch_size = 4

    dataset = SegmentationDataset(phase='test')
    logging.info(f'Test: {len(dataset)}')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet().to(device)
    checkpoint = 'checkpoint/best_model_epoch50_loss0.0249.pth'
    model.load_state_dict(torch.load(checkpoint))
    logging.info(f'Load model from {checkpoint}')

    metrics = StreamSegMetrics(n_classes=2)
    metrics.reset()

    is_save = False
    save_root = 'data/test/pred'
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    model.eval()
    with tqdm(total=len(loader), desc=f'Test') as pbar:
        with torch.no_grad():
            for imgs, masks, img_names in loader:
                imgs = imgs.to(device)
                masks = masks.to(device, dtype=torch.long)
                preds = model(imgs)
                preds = preds.detach().max(dim=1)[1]
                metrics.update(preds, masks)

                if is_save:
                    preds = torch.split(preds, dim=0, split_size_or_sections=1)
                    cmap = colors.ListedColormap(['black', 'white'])
                    norm = colors.Normalize(0, 1)
                    for i in range(len(preds)):
                        img_name = img_names[i]
                        img_path = os.path.join(save_root, img_name)
                        img = preds[i].squeeze().cpu().numpy()
                        img = plt.imshow(img, cmap=cmap, norm=norm)
                        plt.axis('off')
                        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)

                pbar.update(1)
            score = metrics.get_results()
            wandb.log({'metrics': score})
            logging.info(score)


if __name__ == '__main__':
    test()

