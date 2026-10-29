"""Offline Canny inputs and differentiable Sobel supervision (paper Eqs. 5-8)."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def canny_image(image, size=256, low=100, high=200):
    import cv2
    from PIL import Image
    image = image.convert('RGB').resize((size, size), Image.Resampling.BICUBIC)
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    return Image.fromarray(cv2.Canny(gray, low, high))


class SobelContour(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer('kx', torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])[None, None])
        self.register_buffer('ky', self.kx.transpose(-1, -2).clone())

    def forward(self, image):
        image = (image + 1) / 2
        gray = (image * image.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
        gray = F.pad(gray, (1, 1, 1, 1), mode='replicate')
        gx, gy = F.conv2d(gray, self.kx), F.conv2d(gray, self.ky)
        response = (gx.square() + gy.square() + self.eps).sqrt()
        # Per-image maximum normalization keeps responses in [0,1], with a zero baseline.
        response = response - self.eps ** .5
        return response / response.amax(dim=(2, 3), keepdim=True).clamp_min(self.eps)


def contour_loss(image, contour, operator=None, reduction='sum'):
    response = (operator or SobelContour().to(image.device))(image)
    squared = (response - contour).square()
    if reduction == 'sum':
        return squared.flatten(1).sum(1).mean()  # Literal squared L2 in Eq. 7.
    if reduction == 'mean':
        return squared.mean()  # Resolution-independent MSE; documented implementation choice.
    raise ValueError('reduction must be mean or sum')
