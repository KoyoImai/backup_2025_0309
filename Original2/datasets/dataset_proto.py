import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image



class TinyImagenet(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        # print("self.root: ", self.root)
        # num=0
        # print(os.path.join(root, 'tiny-imagenet/processed/x_%s_%02d.npy' %
        #                    ('train' if self.train else 'val', num+1)))

        self.data = []
        for num in range(20):
            self.data.append(np.load(os.path.join(
                root, 'tiny-imagenet/processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        # print("self.data.shape: ", self.data.shape)
        self.data = np.concatenate(np.array(self.data))
        # print("self.data.shape: ", self.data.shape)      # self.data.shape:  (100000, 64, 64, 3)

        self.targets = []
        for num in range(20):
            self.targets.append(np.load(os.path.join(
                root, 'tiny-imagenet/processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.targets = np.concatenate(np.array(self.targets))
        # print("self.targets.shape: ", self.targets.shape)
        # print("self.targets[0:10]: ", self.targets[0:10])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target

