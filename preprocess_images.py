"""Prepare AFAC images with the crop and augmentation used in the notebook."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def build_transform():
    """Apply the notebook's one-time augmentation before training-time resizing."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        # The original crop is offset slightly above the image center.
        transforms.Lambda(lambda image: image.crop((16, 12, 240, 236))),
        transforms.ColorJitter(brightness=0.1, contrast=0.05, saturation=0.1, hue=0.02),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])


def preprocess_directory(input_dir, output_dir, seed=None):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError('Use a separate output directory to preserve the source images.')
    if seed is not None:
        torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = build_transform()
    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    for path in paths:
        with Image.open(path) as image:
            prepared = transform(image.convert('RGB'))
        # Preserve filenames so the original age-annotation file remains valid,
        # including PNG inputs (the notebook converted their extension to JPG).
        transforms.ToPILImage()(prepared).save(output_dir / path.name)
    return len(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, help='Optional seed for repeatable one-time augmentation.')
    args = parser.parse_args()
    count = preprocess_directory(args.input_dir, args.output_dir, args.seed)
    print(f'Prepared {count} images in {args.output_dir}')


if __name__ == '__main__':
    main()
