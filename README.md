# Dog age transformation with a conditional GAN

This repository organizes my contribution to the [KUBIG dog age transformation team project](https://github.com/KU-BIG/KUBIG_2025_FALL/tree/main/KUBIG%20CONTEST/CV_Team6). The project investigated whether a generator conditioned on dog age could produce images showing possible age-related appearance changes. It was a research prototype motivated by missing-dog searches, not a deployed identification tool.

## My contribution

- Led the team project and prepared age-class labels and AFAC image data for experiments.
- Implemented and compared age-conditioned generator and discriminator variants, including condition injection and a projection discriminator.
- Trained a conditional encoder and explored latent optimization to reconstruct an input image using pixelwise and VGG-based perceptual losses before changing its age condition.

The [team README](https://github.com/KU-BIG/KUBIG_2025_FALL/blob/main/KUBIG%20CONTEST/CV_Team6/README.md) gives the broader team context and results. The code here comes from my working notebook. The experiments found that preserving identity and producing convincing age changes remained difficult.

## Code map

| File | Purpose |
| --- | --- |
| `prepare_labels.py` | Convert age-in-months annotations to young/senior labels |
| `data.py` | Load labeled AFAC images |
| `models.py` | Generator, discriminator, encoder, and perceptual model variants |
| `train_gan.py` | Train the condition-injected generator and projection discriminator |
| `train_encoder.py` | Train an image-to-latent encoder on synthetic generator outputs |
| `reconstruct.py` | Optimize a latent code and generate reconstructed/age-switched images |
| `notebooks/dog_age_cgan.ipynb` | Cleaned original research notebook, including exploratory variants |

## Run the stages

Install the packages in `requirements.txt` and provide the AFAC images and annotations separately:

```bash
python prepare_labels.py --annotations DATA/train.txt --images DATA/trainset --output work/AFAC.txt
python train_gan.py --labels work/AFAC.txt --images DATA/trainset --output-dir runs/injected
python train_encoder.py --generator-checkpoint runs/injected/G_epoch_300.pth --output checkpoints/E_best.pth
python reconstruct.py --image DATA/example.jpg --source-age young --generator-checkpoint runs/injected/G_epoch_300.pth --encoder-checkpoint checkpoints/E_best.pth --output-dir results
```

GAN training and latent optimization are computationally expensive. Dataset images, personal dog photographs, trained weights, and generated outputs are omitted. The scripts separate the main stages; the notebook preserves the experimental sequence and alternative architectures.
