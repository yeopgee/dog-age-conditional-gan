"""Train the age-conditioned generator and projection discriminator on AFAC."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from data import AFACDataset
from models import InjectionGenerator, ProjectionDiscriminator


def initialize(module):
    if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        torch.nn.init.normal_(module.weight, 0, 0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(module.weight, 1, 0.02)
        torch.nn.init.zeros_(module.bias)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', required=True, help='Tab-separated filename and Young/Senior label file.')
    parser.add_argument('--images', required=True, help='AFAC image directory.')
    parser.add_argument('--output-dir', default='runs/injected')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(),
                                    transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    loader = DataLoader(AFACDataset(args.labels, args.images, transform=transform),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = InjectionGenerator(z_dim=128, c_dim=2).to(device)
    discriminator = ProjectionDiscriminator(c_dim=2).to(device)
    generator.apply(initialize)
    discriminator.apply(initialize)
    g_opt = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(g_opt, step_size=50, gamma=0.5)
    fixed_z = torch.randn(20, 128, device=device)
    for epoch in range(1, args.epochs + 1):
        for real, labels in loader:
            real, labels = real.to(device), labels.to(device)
            z = torch.randn(len(real), 128, device=device)
            fake_labels = torch.randint(0, 2, (len(real),), device=device)
            condition = F.one_hot(fake_labels, num_classes=2).float()
            fake = generator(z, condition)

            d_opt.zero_grad(set_to_none=True)
            # Hinge losses for real images and age-conditioned fake images.
            d_loss = F.relu(1 - discriminator(real, labels)).mean() + \
                     F.relu(1 + discriminator(fake.detach(), fake_labels)).mean()
            d_loss.backward()
            d_opt.step()

            g_opt.zero_grad(set_to_none=True)
            g_loss = -discriminator(generator(z, condition), fake_labels).mean()
            g_loss.backward()
            g_opt.step()
        scheduler.step()
        print(f'Epoch {epoch}: D={d_loss.item():.4f}, G={g_loss.item():.4f}')
        if epoch % args.save_every == 0:
            # Keep the generator state and training metadata saved by the notebook.
            torch.save({'epoch': epoch, 'G_state_dict': generator.state_dict(),
                        'optimizer_G_state_dict': g_opt.state_dict(),
                        'g_loss': g_loss.item(),
                        'z_sample': z[:20].detach().cpu(),
                        'c_sample': fake_labels[:20].detach().cpu()},
                       output / f'G_epoch_{epoch:03d}.pth')
            with torch.no_grad():
                young = generator(fixed_z, torch.tensor([[1., 0.]], device=device).repeat(len(fixed_z), 1))
                senior = generator(fixed_z, torch.tensor([[0., 1.]], device=device).repeat(len(fixed_z), 1))
                pairs = torch.stack([young, senior], dim=1).flatten(0, 1)
                save_image(((pairs + 1) / 2).clamp(0, 1), output / f'samples_epoch_{epoch:03d}.png', nrow=10)


if __name__ == '__main__':
    main()
