import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from glob import glob
from PIL import Image
import numpy as np

class SlidingWindowTransform:
    def __init__(self, window_size, step_size):
        self.window_size = window_size
        self.step_size = step_size

    def __call__(self, img):
        patches = []
        width, height = img.size
        for i in range(0, width, self.step_size):
            for j in range(0, height, self.step_size):
                right = min(i + self.window_size, width)
                bottom = min(j + self.window_size, height)
                left = right - self.window_size
                top = bottom - self.window_size
                patch = img.crop((left, top, right, bottom))
                patches.append((transforms.ToTensor()(patch), (left, top)))
        return patches

class DIV2K_Loader():
    def __init__(self, batch_size):
        self.data_dir = '/home/cmluser56/data/DIV2K'  # 실제 데이터셋 경로로 변경
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.val_dir = os.path.join(self.data_dir, 'val')
        self.batch_size = batch_size
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

class ImageNet_Loader():
    def __init__(self, batch_size, size, norm=True):
        self.data_dir = '/data/dataset/imagenet'
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.val_dir = os.path.join(self.data_dir, 'val')
        self.batch_size = batch_size
        self.norm = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]] if norm == True else [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop(size*2),
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(self.norm[0], self.norm[1])
            ]),
            'val': transforms.Compose([
                transforms.CenterCrop(size*2),
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(self.norm[0], self.norm[1])
            ]),
        }
    def dataset_load(self):
        image_datasets = {
            'train': datasets.ImageFolder(self.train_dir, self.data_transforms['train']),
            'val': datasets.ImageFolder(self.val_dir, self.data_transforms['val'])
        }
        return image_datasets
    def dataloader(self):
        train_dataset = self.dataset_load()['train']
        val_dataset = self.dataset_load()['val']

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=3),
            'val': DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=3)
        }
        return dataloaders
    
    def raw_dataloader(self):
        image_datasets = {
            'train': datasets.ImageFolder(self.train_dir, transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])),
            'val': datasets.ImageFolder(self.val_dir, transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()]))
        }
        train_dataset = image_datasets['train']
        val_dataset = image_datasets['val']

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=3),
            'val': DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=3)
        }
        return dataloaders

    def dataloader_1k(self, fixed_val = False):
        train_dataset = self.dataset_load()['train']
        val_dataset = self.dataset_load()['val']

        num_train, num_val = len(train_dataset), len(val_dataset)
        subset_train_size, subset_val_size = int(0.1 * num_train), int(0.1 * num_val)

        train_indices = torch.randperm(num_train).tolist()[:subset_train_size]
        if fixed_val == False:
            val_indices = torch.randperm(num_val).tolist()[:subset_val_size]
        else:
            val_indices = torch.arange(num_val).tolist()[:subset_val_size]

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)

        dataloaders = {
            'train': DataLoader(train_subset, batch_size=self.batch_size, shuffle=True,num_workers=3, drop_last=True),
            'val': DataLoader(val_subset, batch_size=self.batch_size, shuffle=False,num_workers=3, drop_last=True)
        }
        return dataloaders        


class Kodak_Patch_Loader(Dataset):
    def __init__(self, patch_size=(128, 128), norm=True):
        self.base_path = '/data2/Kodak'
        self.image_paths = glob(os.path.join(self.base_path, '*.png'))
        self.patch_size = patch_size
        self.patches = self.create_patches_from_images()
        self.num_patches_per_image = self.patches.shape[1]
        self.norm = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]] if norm == True else [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.norm[0], self.norm[1])
        ])

    def create_patches_from_images(self):
        all_patches = []
        for image_path in self.image_paths:
            image = Image.open(image_path).convert('RGB')
            img_width, img_height = image.size
            patch_width, patch_height = self.patch_size

            padded_width = (img_width + patch_width - 1) // patch_width * patch_width
            padded_height = (img_height + patch_height - 1) // patch_height * patch_height

            padded_image = Image.new('RGB', (padded_width, padded_height))
            padded_image.paste(image, (0, 0))

            img_patches = []
            for i in range(0, padded_width, patch_width):
                for j in range(0, padded_height, patch_height):
                    box = (i, j, i + patch_width, j + patch_height)
                    patch = padded_image.crop(box)
                    img_patches.append(np.array(patch))

            img_patches_array = np.array(img_patches)
            all_patches.append(img_patches_array)
        all_patches_array = np.stack(all_patches)
        return all_patches_array

    def __len__(self):
        return self.patches.shape[0] * self.patches.shape[1]

    def __getitem__(self, idx):
        image_idx = idx // self.num_patches_per_image
        patch_idx = idx % self.num_patches_per_image
        patch_uint8 = self.patches[image_idx, patch_idx]
        patch = self.transform(patch_uint8)
        return patch, idx

    def dataloader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=False)

if __name__ == '__main__':
    loader = Kodak_Patch_Loader().dataloader()
    min=torch.ones(3)
    max=torch.zeros(3)
    avg=torch.zeros(3)
    axis=1
    for num_batch, data in enumerate(loader):
        data_origin = data
        data = data.reshape(3, -1)
        min_value = torch.min(data, dim=axis, keepdim=False)[0]
        max_value = torch.max(data, dim=axis, keepdim=False)[0]
        avg_value = torch.mean(data, dim=axis, keepdim=False)
        avg = avg + avg_value
        for idx in range(3):
            if min[idx] >= min_value[idx]: min[idx] = min_value[idx]
            if max[idx] <= max_value[idx]: max[idx] = max_value[idx]
    avg = avg / len(loader)
    print(f'max = {max}, min = {min} avg = {avg}')
