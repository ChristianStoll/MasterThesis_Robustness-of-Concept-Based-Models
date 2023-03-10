# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import Tensor
from typing import List, Tuple


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int, int]]):
        """
        Args:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device: torch.device) -> 'ImageList':
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)

    def requires_grad(self, requires: bool): # TODO my own addition
        cast_tensor = self.tensors.requires_grad()
        return ImageList(cast_tensor, self.image_sizes)
