from typing import Optional
from torch import Tensor
from typing import List

import torchvision
import torch

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
    
def _max_by_axis(the_list):
    """
    Compare image sizes and return the maximum size that covers all images.
    The idea is borrowed from facebookresearch/detr repository.
    """
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    """
    Function to create a NestedTensor from a list of images (tensors).
    The idea is borrowed from facebookresearch/detr repository.
    """
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    """
    A wrapper around torch.nn.functional.interpolate to handle NestedTensor.
    """
    if isinstance(input, NestedTensor):
        input = input.tensors
    return torchvision.ops.misc.interpolate(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

def get_world_size():
    """
    Get the world size for distributed training.
    """
    if not torch.distributed.is_available():
        return 1
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def is_dist_avail_and_initialized():
    """
    Check if distributed training is available and initialized.
    """
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """
    Compute the accuracy over the k top predictions for the specified values of k.
    """
    if target.numel() == 0:
        return [torch.tensor(0.0, device=output.device) for _ in topk]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
