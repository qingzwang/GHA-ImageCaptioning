import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import numpy as np
import nltk
from PIL import Image
import pickle


class CocoDataset(data.Dataset):
    def __init__(self, image_path, split, width=224, height=224,transformer=None):
        self._image_path = image_path   # dir of images
        self._split = split
        self._width = width
        self._height = height
        self._transformer = transformer     # torch image transformer, e.g. crop, color distortion
        self._totensor = transforms.ToTensor()
        self._normalizer = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __getitem__(self, index):
        """
        1. read one data from file (e.g. images)
        2. preprocess the data (e.g. crop, resize)
        3. return one data pair (e.g. image and label)
        :param index:
        :return:
        """
        image_file = os.path.join(self._image_path, self._split[index]['filename'])
        image_id = self._split[index]['image_id']
        caption = self._split[index]['caption_id']
        image = Image.open(image_file).convert('RGB')
        if self._transformer is not None:
            image = self._transformer(image)
        else:
            image = image.resize((self._width, self._height), resample=Image.BILINEAR)
        image = self._totensor(image)
        image = self._normalizer(image)

        return torch.IntTensor([image_id]), image, torch.IntTensor(caption)

    def __len__(self):
        return len(self._split)


def collate_fn(data):
    """
    creat mini-batch tensors from the list of tuples (image, caption)
    :param data:
    list of tuples (image_id, image, caption)
    -image: torch tensor of shape (3, width, height)
    -caption: torch tensor of shape (?)
    -image_id: torch tensor of shape (1)
    :return:
    images: torch tensor of shape (batch_size, 3, width, height)
    inputs: torch tensor of shape (batch_size, length)
    targets: torch tensor of shape (batch_size, length)
    mask: torch tensor of shape (batch_size, length)
    image_ids: torch tensor of shape (batch_size, 1)
    """
    image_ids, images, captions = zip(*data)

    image_ids = torch.stack(image_ids, 0)   # (batch_size, 3, width, height)
    images = torch.stack(images, 0)     # (batch_size, 1)

    lengths = [len(cap) for cap in captions]
    mask = torch.zeros(len(captions), max(lengths)-1)
    inputs = torch.zeros(len(captions), max(lengths)-1).long()
    targets = torch.zeros(len(captions), max(lengths) - 1).long()
    for i, cap in enumerate(captions):
        end = lengths[i] - 1
        inputs[i, :end] = cap[:end]
        targets[i, :end] = cap[1:]
        mask[i, :end] = torch.ones(end)

    return image_ids, images, inputs, targets, mask


def get_data_loader(image_dir, split_file,
                    transformer, split_key='train', width=224, height=224,
                    batch_size=10, num_workers=6, shuffle=True):

    with open(split_file, 'r') as f:
        split = pickle.load(f)
    if isinstance(split_key, list):
        image_cap = []
        for key in split_key:
            image_cap.extend(split[key])
    else:
        image_cap = split[split_key]
    coco = CocoDataset(
        image_path=image_dir,
        split=image_cap,
        transformer=transformer,
        width=width,
        height=height)
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)

    return data_loader
