import os

from torchvision import datasets,transforms
from torch.utils.data import DataLoader, Dataset



from torchvision import datasets,transforms
from torch.utils.data import DataLoader, Dataset
import cv2
class ImageMaskDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        
        image = cv2.imread(self.image_paths[index])
        mask =  cv2.imread(self.mask_paths[index])

        if self.transform is not None:
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image=image_np)
            image = image['image']
            mask_np = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = self.transform(image=mask_np)
            mask = mask['image']

        return image, mask




def create_dataloader_for_mask(
        train_dir: str,
        mask_dir: str,
        transform,
        batch_size: int,
        num_workers: int = 12,
):
    data=ImageMaskDataset(train_dir,mask_dir,transform)
    dataloader=DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,

    )

    return dataloader


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        label = self.labels[index]
        image = cv2.imread(self.images[index])

        if self.transform is not None:
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image=image_np)
            image = image['image']

        return image, label

def create_dataloader_images(
        X_train_dir: str,
        y_train,
        X_test_dir: str,
        y_test,
        test_transform,
        train_transform,
        batch_size: int,
        num_workers: int ,
):
    train_data=CustomDataset(X_train_dir,y_train,transform=train_transform)
    test_data=CustomDataset(X_test_dir,y_test,transform=test_transform)

    train_dataloader=DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader=DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader,test_dataloader