# 48kHz GAN-Based Neural Vocoder

A high-fidelity, GAN-based neural vocoder designed for 48kHz audio synthesis. This model takes mel spectrograms as input and generates high-quality waveforms with efficient and realistic speech synthesis.

Features

Uses GAN-based architecture for high-quality waveform generation.

Supports 48kHz audio output for improved fidelity.

Implements Multi-Scale and Multi-Period Discriminators for enhanced realism.

EMA (Exponential Moving Average) generator for smoother training convergence.

Includes gradient clipping and TF32 optimizations for stable training on NVIDIA GPUs.



# Model Architecture

Generator: Uses transposed convolutions and residual blocks for upsampling mel spectrograms to waveforms.

Discriminators:

Multi-Scale Discriminator (MSD): Analyzes different scales of waveforms.

Multi-Period Discriminator (MPD): Focuses on periodic patterns in speech.

Losses:

Feature matching loss

Adversarial loss for generator & discriminator

L1 loss for waveform reconstruction



# Checkpoints & Logs

Checkpoints are saved in vocoder_checkpoints/

Training logs (loss, gradient values) are stored in loss.json and Gradient_value.json.



# Future Improvements

Experimenting with different GAN architectures for better voice clarity.

Optimizing inference speed with TensorRT or ONNX.

Fine-tune model for voice cloning tasks.

Will be uploading Inference code and trained checkpoints once it is trained.



# Acknowledgments

Inspired by HiFi-GAN and MelGAN architectures. Built with PyTorch, torchaudio, and numpy.



**Copyright (c) 2025 [PhoenixtBlaze]**

All rights reserved. This software and its source code are proprietary and may not be copied, modified, distributed, or used in any form without explicit permission from the copyright holder. Unauthorized use is strictly prohibited.
