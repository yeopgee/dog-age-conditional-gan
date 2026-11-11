"""Train the encoder on a fixed synthetic dataset, as in the project notebook."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import ConcatEncoder, InjectionGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generator-checkpoint', required=True)
    parser.add_argument('--output', default='checkpoints/encoder_concat/E_best.pth')
    parser.add_argument('--last-output', default='checkpoints/encoder_concat/E_last.pth')
    parser.add_argument('--samples', type=int, default=100_000)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = InjectionGenerator(z_dim=128, c_dim=2).to(device)
    checkpoint = torch.load(args.generator_checkpoint, map_location=device)
    generator.load_state_dict(checkpoint['G_state_dict'])
    generator.eval().requires_grad_(False)
    encoder = ConcatEncoder(z_dim=128, c_dim=2).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    best_output, last_output = Path(args.output), Path(args.last_output)
    best_output.parent.mkdir(parents=True, exist_ok=True)
    last_output.parent.mkdir(parents=True, exist_ok=True)

    # The notebook samples 100,000 latent/age pairs once and revisits them for
    # 60 epochs. Keep those pairs fixed, generating their images batch by batch
    # to avoid holding all 100,000 full-resolution images in memory.
    latents = torch.randn(args.samples, 128)
    labels = torch.randint(0, 2, (args.samples,))
    conditions = F.one_hot(labels, num_classes=2).float()
    loader = DataLoader(TensorDataset(latents, conditions),
                        batch_size=args.batch_size, shuffle=True)

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        total_loss = 0.0
        for latent_cpu, condition_cpu in loader:
            latent = latent_cpu.to(device)
            condition = condition_cpu.to(device)
            with torch.no_grad():
                image = generator(latent, condition)

            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(encoder(image, condition), latent)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(latent)

        average_loss = total_loss / args.samples
        print(f'Epoch {epoch:02d}: latent MSE {average_loss:.6f}')
        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(encoder.state_dict(), best_output)

    torch.save(encoder.state_dict(), last_output)
    print(f'Saved best encoder to {best_output} and final encoder to {last_output}')


if __name__ == '__main__':
    main()
