import numpy as np
import torch
import PIL
from PIL import Image
from torchvision.transforms import functional as F


def image2tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch
    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type
    Returns:
        tensor (torch.Tensor): Data types supported by PyTorch
    """
    # Convert image data type to Tensor data type
    tensor = F.to_tensor(image)

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor

def preprocess(images, channel_order='RGB'):
    """Preprocesses the input images if needed.
    This function assumes the input numpy array is with shape [batch_size,
    height, width, channel]. Here, `channel = 3` for color image and
    `channel = 1` for grayscale image. The returned images are with shape
    [batch_size, channel, height, width].
    NOTE: The channel order of input images is always assumed as `RGB`.
    Args:
      images: The raw inputs with dtype `numpy.uint8` and range [0, 255].
    Returns:
      The preprocessed images with dtype `numpy.float32` and range
        [0, 1].
    """
    # input : numpy, np.uint8, 0~255, RGB, BHWC
    # output : numpy, np.float32, 0~1, RGB, BCHW

    image_channels = 3
    max_val = 1.0
    min_val = 0.0

    if image_channels == 3 and channel_order == 'BGR':
      images = images[:, :, :, ::-1]
    images = images / 255.0 * (max_val - min_val) + min_val
    images = images.astype(np.float32).transpose(0, 3, 1, 2)
    return images

def postprocess(images, channel=3):
    """Post-processes images from `torch.Tensor` to `numpy.ndarray`."""
    # input : tensor, 0~1, RGB, BCHW
    # output : np.uint8, 0~255, BGR, BHWC

    images = images.detach().cpu().numpy()
    images = images * 255.
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    if channel==1:
      images = images.transpose(0, 2, 3, 1)
    else:
      images = images.transpose(0, 2, 3, 1)[:,:,:,[2,1,0]]
    return images

def Lanczos_resizing(image_target, resizing_tuple=(256,256)):
    # input : 0~1, RGB, BCHW, Tensor
    # output : 0~1, RGB, BCHW, Tensor
    image_target_resized = image_target.clone().cpu().numpy()
    image_target_resized = image_target_resized * 255.
    image_target_resized = np.clip(image_target_resized + 0.5, 0, 255).astype(np.uint8)

    image_target_resized = image_target_resized.transpose(0, 2, 3, 1)
    tmps = []
    for i in range(image_target_resized.shape[0]):
        tmp = image_target_resized[i]
        tmp = Image.fromarray(tmp) # PIL, 0~255, uint8, RGB, HWC
        tmp = np.array(tmp.resize(resizing_tuple, PIL.Image.LANCZOS))
        tmp = torch.from_numpy(preprocess(tmp[np.newaxis,:])).cuda()
        tmps.append(tmp)
    return torch.cat(tmps, dim=0)

def ganimage_preprocess(images):
    #input : -1~1, RGB, BCHW, Tensor
    #output : 0~1, RGB, BCHW, Tensor
    images = (images+1.)/2.
    return images
