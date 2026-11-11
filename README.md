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

- **Generator:** The paper supplies age at the generator input. I embedded the age condition and injected it into intermediate features to strengthen its effect.
- **Discriminator:** The paper conditions an early convolutional layer. In our initial implementation, spatially broadcasting the age condition was associated with overfitting and mode collapse. I tried both a projection discriminator and a variant that concatenated a learned age embedding with convolutional image features before the fully connected layer.
- **Encoder:** Our initial version spatially concatenated age with the image and was unstable. I instead extracted image features with convolutions and concatenated a linear age embedding before the fully connected layer.
- **Latent-vector optimization:** The paper uses a FaceNet-based identity loss. Because that face-recognition model does not apply to dogs, I optimized the latent vector with pixelwise and VGG-feature perceptual losses before switching to the target age.

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
