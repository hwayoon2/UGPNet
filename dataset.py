# code heavily borrowed from https://github.com/Lornatang/ESRGAN-PyTorch/blob/2f7f58290e6d84fdaad60f72822cb91139f280c5/dataset.py#L190

import os
import queue
import threading

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from image_tools import image2tensor
import math
import random as rand
from torchvision.transforms.functional import normalize
import torchvision.transforms as transforms
from collections import OrderedDict

__all__ = [
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
    "PairedImageDataset"
]


class PairedImageDataset(Dataset):
    """Define training/valid dataset loading methods.
    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): High resolution image size.
        upscale_factor (int): Image up scale factor.
        mode (str): Data set loading method, the training data set is for data enhancement, and the
            verification dataset is not for data enhancement.
    """

    def __init__(self, image_lq_dir: str, image_gt_dir: str) -> None:
        super(PairedImageDataset, self).__init__()
        self.image_file_names = [os.path.join(image_lq_dir, image_file_name) for image_file_name in os.listdir(image_lq_dir)]
        self.image_gt_dir = image_gt_dir

    def __getitem__(self, batch_index: int):
        # Read a batch of image data
        lq_image = cv2.imread(self.image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        image_basename = os.path.basename(self.image_file_names[batch_index])
        gt_image = cv2.imread(os.path.join(self.image_gt_dir, image_basename), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        # BGR convert to RGB
        lq_image = cv2.cvtColor(lq_image, cv2.COLOR_BGR2RGB)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]

        lq_tensor = image2tensor(lq_image, range_norm=False, half=False)
        gt_tensor = image2tensor(gt_image, range_norm=False, half=False)

        return {"lq": lq_tensor, "gt": gt_tensor, "name": image_basename[:-4]}

    def __len__(self) -> int:
        return len(self.image_file_names)


class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.
    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """A fast data prefetch dataloader.
    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)