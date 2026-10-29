"""Caption metadata and DAMSM vocabulary, with paired offline contours."""
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from .datasets import TextImgDataset
from .contours import canny_image


class ContourDataset(TextImgDataset):
    def __init__(self, args, split):
        # Keep original metadata parsing and vocabulary; paper resizes full images.
        super().__init__(split=split, transform=None, args=args)
        self.args = args
        self.contour_root = Path(args.contour_dir) / split
        if not self.filenames:
            raise FileNotFoundError(f'No filenames in {args.data_dir}/{split}/filenames.pickle')

    def load_bbox(self):
        return None  # Paper preprocessing uses the full image, resized to 256 x 256.

    def image_path(self, key):
        root = Path(self.data_dir)
        if self.dataset_name == 'birds':
            return root / 'CUB_200_2011/images' / f'{key}.jpg'
        folder = 'train2014' if self.split == 'train' else 'val2014'
        return root / 'images' / folder / f'{key}.jpg'

    def __getitem__(self, index):
        key = self.filenames[index]
        image = Image.open(self.image_path(key)).convert('RGB').resize(
            (self.args.imsize, self.args.imsize), Image.Resampling.BICUBIC)
        contour_path = self.contour_root / f'{key}.png'
        if not contour_path.exists():
            raise FileNotFoundError(f'{contour_path}: run preprocess first')
        contour = Image.open(contour_path).convert('L')
        if contour.size != image.size:
            raise ValueError(f'Contour size mismatch: {contour_path}; regenerate contours')
        sentence = np.random.randint(self.embeddings_num) if self.split == 'train' else 0
        caps, length = self.get_caption(index * self.embeddings_num + sentence)
        image = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 127.5 - 1
        contour = torch.from_numpy(np.asarray(contour).copy())[None].float() / 255
        return image, torch.from_numpy(caps[:, 0]), length, contour, key


def preprocess(args):
    from tqdm import tqdm
    for split in ('train', 'test'):
        ds = ContourDataset(args, split)
        for key in tqdm(ds.filenames, desc=f'Canny {split}'):
            target = ds.contour_root / f'{key}.png'
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(ds.image_path(key)) as image:
                canny_image(image, args.imsize, args.canny_low, args.canny_high).save(target)
    root = Path(args.contour_dir)
    (root / 'preprocess.json').write_text(json.dumps({
        'imsize': args.imsize, 'low': args.canny_low, 'high': args.canny_high,
        'resize': 'bicubic', 'crop': False, 'gray': 'RGB2GRAY'}, indent=2))


def encode_text(encoder, captions, lengths):
    device = next(encoder.parameters()).device
    lengths, order = lengths.sort(descending=True)
    with torch.no_grad():
        _, sentence = encoder(captions[order].to(device), lengths.cpu(), encoder.init_hidden(len(order)))
    return sentence[order.argsort().to(device)].detach()
