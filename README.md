# Dog age transformation with an age-conditional GAN

This repository documents my contribution to the [KUBIG dog age transformation team project](https://github.com/KU-BIG/KUBIG_2025_FALL/tree/main/KUBIG%20CONTEST/CV_Team6). We explored whether a model could transform a dog photograph to suggest an older appearance, motivated by searches for missing dogs. It is an experimental image-transformation project.

## Motivation

A conditional generator can create an image for a requested age category, but transforming a *particular* dog also requires reconstructing that dog's appearance. The project therefore combined an age-conditional GAN with an encoder and latent-vector optimization.

## My contribution

1. Prepared Young and Senior labels from the AFAC age annotations, excluding the ambiguous middle-age range for these experiments.
2. Implemented and compared age-conditioned generator and discriminator variants, including condition injection and a projection discriminator.
3. Trained an encoder to initialize the latent vector. Then optimized that vector to reconstruct an input photograph using pixelwise and VGG-based perceptual losses before changing the age condition.

```text
Input photo + source age
        │
        ▼
Encoder → initial latent vector
        │
        ▼
Latent optimization → reconstruction under source age
        │
        ▼
Same latent vector + target age → transformed image
```

## Results

The examples below show the original photograph, its reconstruction, and the output under the Senior condition.

![First dog age-transformation example: original, reconstruction, transformed](assets/reconstruction-example-1.png)

![Second dog age-transformation example: original, reconstruction, transformed](assets/reconstruction-example-2.png)

*Examples from my KUBIG presentation. They illustrate the reconstruction and transformation pipeline, including its visible limitations.*

## Discussion

The experiments exposed a difficult tradeoff: changing the age condition often changed other aspects of the dog's appearance. Preserving identity and generating convincing age-related changes remained challenging, especially with noisy age labels and unstable GAN training. The [team README](https://github.com/KU-BIG/KUBIG_2025_FALL/blob/main/KUBIG%20CONTEST/CV_Team6/README.md) describes the broader project and other variants.

## Code and usage

| File | Role |
| --- | --- |
| [`prepare_labels.py`](prepare_labels.py) | Convert age annotations to two classes |
| [`models.py`](models.py) | Generator, discriminator, encoder, and perceptual features |
| [`train_gan.py`](train_gan.py) | Train the conditional GAN |
| [`train_encoder.py`](train_encoder.py) | Train the latent initializer |
| [`reconstruct.py`](reconstruct.py) | Optimize a latent vector and render age-conditioned output |
| [`notebooks/dog_age_cgan.ipynb`](notebooks/dog_age_cgan.ipynb) | Exploratory experiments and model variants |

With the AFAC data available and dependencies installed from [`requirements.txt`](requirements.txt), the stages are:

```bash
python prepare_labels.py --annotations DATA/train.txt --images DATA/trainset --output work/AFAC.txt
python train_gan.py --labels work/AFAC.txt --images DATA/trainset --output-dir runs/injected
python train_encoder.py --generator-checkpoint runs/injected/G_epoch_300.pth --output checkpoints/E_best.pth
python reconstruct.py --image DATA/example.jpg --source-age young --generator-checkpoint runs/injected/G_epoch_300.pth --encoder-checkpoint checkpoints/E_best.pth --output-dir results
```

The dataset and trained weights are not redistributed in this repository.
