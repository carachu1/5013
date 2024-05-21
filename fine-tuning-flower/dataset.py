from PIL import Image
import torch
from torch.utils.data import Dataset
import random

class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, images_path, images_class, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

        # Preload all images to check their modes
        self.valid_indices = [i for i, path in enumerate(self.images_path) if Image.open(path).mode == 'RGB']

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, item):
        # Ensure we use only valid indices
        idx = self.valid_indices[item]
        img_path = self.images_path[idx]
        label = self.images_class[idx]

        img = Image.open(img_path)
        if img.mode != 'RGB':
            # This case should never happen due to preloading, but we check again
            idx = random.choice(self.valid_indices)
            img_path = self.images_path[idx]
            img = Image.open(img_path)
        label = self.images_class[idx]

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

# class MyDataSet(torch.utils.data.Dataset):
#     def __init__(self, images_path, images_class, transform=None):
#         self.images_path = images_path
#         self.images_class = images_class
#         self.transform = transform

#     def __len__(self):
#         return len(self.images_path)

#     def __getitem__(self, item):
#         img = Image.open(self.images_path[item])
#         if img.mode != 'RGB':
#             # img = img.convert('RGB')
#             # print(f"Converted image {self.images_path[item]} to RGB mode")
#             idx = random.choice(self.valid_indices)
#             img_path = self.images_path[idx]
#             # label = self.images_class[idx]
#             img = Image.open(img_path)
        
#         label = self.images_class[item]
        
#         if self.transform is not None:
#             img = self.transform(img)
        
#         return img, label

#     @staticmethod
#     def collate_fn(batch):
#         images, labels = tuple(zip(*batch))
#         images = torch.stack(images, dim=0)
#         labels = torch.as_tensor(labels)
#         return images, labels
