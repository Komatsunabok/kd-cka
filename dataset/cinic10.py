from __future__ import print_function

import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from PIL import Image

def get_data_folder(data_folder_dir='../data/cinic-10'):
    """
    Return the path to store the data
    """
    data_folder = data_folder_dir
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder

class CINIC10BackCompat(ImageFolder):
    """
    CINIC10Instance+Sample Dataset
    """

    @property
    def train_labels(self):
        return self.targets

    @property
    def test_labels(self):
        return self.targets

    @property
    def train_data(self):
        return self.data

    @property
    def test_data(self):
        return self.data

class CINIC10Instance(CINIC10BackCompat):
    """CINIC-10 Dataset with instance index for contrastive learning."""
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


def get_cinic10_dataloaders(batch_size=128, num_workers=8, is_instance=False, data_folder_dir="../data/cinic-10"):
    """
    CINIC-10 DataLoader
    """
    print("CINIC-10 DataLoader")
    data_folder = get_data_folder(data_folder_dir)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)),
    ])

    if is_instance:
        train_set = CINIC10Instance(root=os.path.join(data_folder, 'train'),
                                            transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.ImageFolder(root=os.path.join(data_folder, 'train'),
                                            transform=train_transform)
        # train_set = datasets.CINIC10(root=data_folder,
        #                              download=True,
        #                              train=True,
        #                              transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.ImageFolder(root=os.path.join(data_folder, 'test'),
                                    transform=test_transform)
    # test_set = datasets.CINIC10(root=data_folder,
    #                             download=True,
    #                             train=False,
    #                             transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader

class CINIC10InstanceSample(CINIC10BackCompat):
    """
    CINIC10Instance+Sample Dataset
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 10
        num_samples = len(self.data)
        label = self.targets

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # Convert to PIL Image for consistency
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # Directly return
            return img, target, index
        else:
            # Sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx

def get_cinic10_dataloaders_sample(batch_size=128, num_workers=8, k=4096, mode='exact',
                                   is_sample=True, percent=1.0):
    """
    CINIC-10 DataLoader with sampling
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)),
    ])

    train_set = CINIC10InstanceSample(root=os.path.join(data_folder, 'train'),
                                      download=True,
                                      train=True,
                                      transform=train_transform,
                                      k=k,
                                      mode=mode,
                                      is_sample=is_sample,
                                      percent=percent)
    n_data = len(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.ImageFolder(root=os.path.join(data_folder, 'test'),
                                    transform=test_transform)
    # test_set = datasets.CINIC10(root=data_folder,
    #                             download=True,
    #                             train=False,
    #                             transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, test_loader, n_data