# Dog age transformation with an age-conditional GAN

This repository contains my age-conditional GAN experiments from the [KUBIG dog age transformation team project](https://github.com/KU-BIG/KUBIG_2025_FALL/tree/main/KUBIG%20CONTEST/CV_Team6). For more information about the project, see the team repository.

## Motivation

The goal was to show how a missing dog might look as it aged. This requires changing the apparent age of a photograph while preserving the individual dog's appearance.

## Data

- [AFAC 2023 pet-age recognition dataset](https://aistudio.baidu.com/datasetdetail/225418/0): age annotations used to prepare the Young and Senior conditions.

## My contribution

- Prepared Young and Senior labels from AFAC age annotations, excluding the middle-age range for these experiments.

I adapted the face-aging pipeline of [Antipov, Baccouche, and Dugelay's Age-cGAN (2017)](https://arxiv.org/html/1702.01983v2) to dog photographs. The figure shows the **original paper's pipeline**; the changes below describe my implementation.

![Original Age-cGAN pipeline: encoder, latent-vector optimization, and generation at a target age](assets/age-cgan-paper-figure-1.png)

*Figure 1 from [Antipov et al. (2017)](https://arxiv.org/html/1702.01983v2). The figure depicts face aging, not the modified dog model.*

### 1. Generator

- **Paper:** Age-cGAN adds the age condition at the generator input.
- **Our method:** I embedded the age condition and injected it into intermediate generator features to make age control more effective.

### 2. Discriminator

- **Paper:** Age-cGAN injects the age condition at the first convolutional layer.
- **Our method:** Our initial version broadcast the age condition across the image; it overfit to age and showed mode collapse. I then tested two alternatives: a projection discriminator and a discriminator that concatenates an age embedding with extracted image features before the fully connected layer.

### 3. Encoder

- **Paper:** The encoder estimates an initial latent vector from the face image, E(x). The age condition is supplied separately to the generator.
- **Our method:** We first tried concatenating a spatial age map with the dog image before encoding, but this version was unstable. I then extracted image features with convolutions and concatenated a learned age embedding before the final fully connected layer.

### 4. Latent-vector optimization

- **Paper:** Age-cGAN optimizes the latent vector using a FaceNet-based identity loss to preserve the person's face.
- **Our method:** FaceNet does not provide a suitable identity measure for dogs, so I optimized the latent vector using pixelwise reconstruction and VGG-feature perceptual losses before switching to the target age.

## Results

The examples below show the original photograph, its reconstruction, and the output under the Senior condition.

![First dog age-transformation example: original, reconstruction, transformed](assets/reconstruction-example-1.png)

![Second dog age-transformation example: original, reconstruction, transformed](assets/reconstruction-example-2.png)

![Third dog age-transformation example: original, reconstruction, transformed](assets/reconstruction-example-3.png)

*Examples from my KUBIG presentation.*

## Limitations

- **Model:** GAN training can be unstable and susceptible to mode collapse; its image quality can lag behind diffusion-based methods.
- **Data:** Unlike the roughly 120,000 mostly frontal human faces used in Age-cGAN, the dog photos varied in breed, pose, appearance, and background, with only about 16,000 small-dog images available.
- **Age condition:** Dogs show fewer obvious aging cues than people, so changing the age condition often had a limited visible effect.

## Code structure

| File | Role |
| --- | --- |
| [`prepare_labels.py`](prepare_labels.py) | Convert age annotations to two classes |
| [`models.py`](models.py) | Generator, discriminator, encoder, and perceptual features |
| [`train_gan.py`](train_gan.py) | Train the conditional GAN |
| [`train_encoder.py`](train_encoder.py) | Train the latent initializer |
| [`reconstruct.py`](reconstruct.py) | Optimize a latent vector and render age-conditioned output |
| [`notebooks/dog_age_cgan.ipynb`](notebooks/dog_age_cgan.ipynb) | Exploratory experiments and model variants |
