"""Train an encoder to recover latent codes from generated images."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from models import ConcatEncoder, InjectionGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generator-checkpoint', required=True)
    parser.add_argument('--output', default='checkpoints/E_best.pth')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = InjectionGenerator(z_dim=128, c_dim=2).to(device)
    checkpoint = torch.load(args.generator_checkpoint, map_location=device)
    generator.load_state_dict(checkpoint['G_state_dict'])
    generator.eval().requires_grad_(False)
    encoder = ConcatEncoder(z_dim=128, c_dim=2).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    for step in range(1, args.steps + 1):
        z = torch.randn(args.batch_size, 128, device=device)
        labels = torch.randint(0, 2, (args.batch_size,), device=device)
        condition = F.one_hot(labels, num_classes=2).float()
        with torch.no_grad():
            images = generator(z, condition)
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(encoder(images, condition), z)
        loss.backward()
        optimizer.step()
        if loss.item() < best:
            best = loss.item()
            torch.save(encoder.state_dict(), output)
        if step % 100 == 0:
            print(f'Step {step}: latent MSE {loss.item():.5f}')


if __name__ == '__main__':
    main()
