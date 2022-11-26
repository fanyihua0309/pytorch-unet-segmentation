import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random


class SegmentationDataset(Dataset):
    def __init__(self, root='data', phase='train'):
        self.phase = phase
        imgs_dir = os.path.join(root, phase, 'imgs')
        masks_dir = os.path.join(root, phase, 'masks')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])
        img_names, imgs, masks = [], [], []
        for img_name in os.listdir(imgs_dir):
            img_names.append(img_name)
            img_path = os.path.join(imgs_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            imgs.append(img)
            mask_path = os.path.join(masks_dir, img_name)
            mask = Image.open(mask_path).convert('L')
            masks.append(mask)
        self.img_names = img_names
        self.imgs = imgs
        self.masks = masks

    def __getitem__(self, index):
        img = self.imgs[index]
        mask = self.masks[index]
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
            mask = mask.squeeze()
        if self.phase == 'train':
            return img, mask
        else:
            img_name = self.img_names[index]
            return img, mask, img_name

    def __len__(self):
        return len(self.imgs)


def train_test_split(root='data', test_ratio=0.1):
    imgs_dir = os.path.join(root, 'train', 'imgs')
    total = os.listdir(imgs_dir)
    test_num = int(len(total) * test_ratio)
    print(f'Total: {len(total)}   Train: {len(total) - test_num}   Test: {test_num}')
    test = random.sample(total, test_num)
    for img_name in test:
        img_path = os.path.join(root, 'train', 'imgs', img_name)
        test_img_path = os.path.join(root, 'test', 'imgs', img_name)
        os.rename(img_path, test_img_path)
        mask_path = os.path.join(root, 'train', 'masks', img_name)
        test_mask_path = os.path.join(root, 'test', 'masks', img_name)
        os.rename(mask_path, test_mask_path)
    print('Done split train and test dataset.')


if __name__ == '__main__':
    train_test_split()

