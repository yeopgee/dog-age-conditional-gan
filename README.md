# Dog age transformation with an age-conditional GAN

This repository contains my age-conditional GAN experiments from the [KUBIG dog age transformation team project](https://github.com/KU-BIG/KUBIG_2025_FALL/tree/main/KUBIG%20CONTEST/CV_Team6). For more information about the project, see the team repository.

## Motivation

The goal was to show how a missing dog might look as it aged. This requires changing the apparent age of a photograph while preserving the individual dog's appearance.

## My contribution

- Prepared Young and Senior labels from AFAC age annotations, excluding the middle-age range for these experiments.
- Implemented age-conditioned generator and discriminator variants, including condition injection and a projection discriminator.
- Trained an encoder to initialize the latent vector, then optimized that vector with pixelwise and VGG-based perceptual losses to reconstruct the input before changing the age condition.

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

*Examples from my KUBIG presentation: original photograph, reconstruction, and output under the Senior condition.*

The model produced age-conditioned outputs, but the examples also show changes in the dog's appearance beyond age.

## Discussion

The main challenge was preserving the dog's identity while changing only its apparent age. Noisy age labels and variation in breed, pose, and background made that separation difficult.

## Code structure

| File | Role |
| --- | --- |
| [`prepare_labels.py`](prepare_labels.py) | Convert age annotations to two classes |
| [`models.py`](models.py) | Generator, discriminator, encoder, and perceptual features |
| [`train_gan.py`](train_gan.py) | Train the conditional GAN |
| [`train_encoder.py`](train_encoder.py) | Train the latent initializer |
| [`reconstruct.py`](reconstruct.py) | Optimize a latent vector and render age-conditioned output |
| [`notebooks/dog_age_cgan.ipynb`](notebooks/dog_age_cgan.ipynb) | Exploratory experiments and model variants |
