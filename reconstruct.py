"""Reconstruct an input dog image and render it under the other age condition."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

from models import ConcatEncoder, InjectionGenerator, VGGFeatureExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--source-age', choices=['young', 'senior'], required=True)
    parser.add_argument('--generator-checkpoint', required=True)
    parser.add_argument('--encoder-checkpoint', required=True)
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--steps', type=int, default=1000)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(),
                                    transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    image = transform(Image.open(args.image).convert('RGB')).unsqueeze(0).to(device)
    condition = torch.tensor([[1., 0.]] if args.source_age == 'young' else [[0., 1.]], device=device)
    generator = InjectionGenerator(z_dim=128, c_dim=2).to(device)
    checkpoint = torch.load(args.generator_checkpoint, map_location=device)
    generator.load_state_dict(checkpoint['G_state_dict'])
    generator.eval().requires_grad_(False)
    encoder = ConcatEncoder(z_dim=128, c_dim=2).to(device)
    encoder.load_state_dict(torch.load(args.encoder_checkpoint, map_location=device))
    encoder.eval().requires_grad_(False)
    perceptual = VGGFeatureExtractor().to(device)
    perceptual.eval().requires_grad_(False)
    with torch.no_grad():
        initial_z = encoder(image, condition)
        target_features = perceptual((image + 1) / 2)
    z = initial_z.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=0.01)
    best = float('inf')
    best_z = z.detach().clone()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        reconstruction = generator(z, condition)
        pixel_loss = F.mse_loss(reconstruction, image)
        feature_loss = F.l1_loss(perceptual((reconstruction + 1) / 2), target_features)
        loss = pixel_loss + 0.05 * feature_loss
        loss.backward()
        optimizer.step()
        if loss.item() < best:
            best, best_z = loss.item(), z.detach().clone()
        if (step + 1) % 100 == 0:
            print(f'Step {step + 1}: reconstruction loss {loss.item():.5f}')
    with torch.no_grad():
        reconstruction = generator(best_z, condition)
        transformed = generator(best_z, 1 - condition)
    for name, tensor in [('input', image), ('reconstruction', reconstruction), ('transformed', transformed)]:
        save_image(((tensor + 1) / 2).clamp(0, 1), output / f'{name}.png')


if __name__ == '__main__':
    main()
