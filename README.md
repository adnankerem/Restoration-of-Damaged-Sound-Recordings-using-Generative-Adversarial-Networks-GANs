# Restoration of Damaged Sound Recordings using Generative Adversarial Networks (GANs)

## Overview

This repository contains the implementation and results of the project: **Restoration of Damaged Sound Recordings Using GANs**, conducted as part of a Master's thesis at the University of Padova.

The project explores the application of Generative Adversarial Networks (GANs) in restoring and enhancing corrupted audio signals. It uses techniques like **MFCCs** (Mel-Frequency Cepstral Coefficients) and **STFT** (Short-Time Fourier Transform) for audio preprocessing. The repository includes the full pipeline for dataset preparation, model training, audio reconstruction, and evaluation.

The thesis further elaborates on the challenges of audio restoration, the design of GAN architectures tailored for spectrogram data, and the nuances of audio reconstruction techniques.

## Features

- Implementation of **Deep Convolutional GAN (DCGAN)** and its variants.
- **MFCC** and **STFT**-based preprocessing for training.
- Logarithmic scaling for better dynamic range handling.
- **Adversarial loss and content loss (MSE)** for improved learning.
- Custom datasets for clean and noisy audio samples.
- Detailed evaluation metrics: MSE, RMSE, SNR, LSD, STOI.
- Comprehensive visualization tools for spectrograms and audio quality.

## Repository Structure

```plaintext
.
├── README.md             # Project documentation (this file)
├── requirements.txt      # Python dependencies
├── scripts               # Main code directory
│   ├── Training_DCGAN.py           # Training script for GANs
│   ├── Preprocess_Afterprocess_GAN_Combined.py  # Pre- and post-processing code
│   ├── Dataloader_Prepare_And_Save.py           # Script to prepare and save data
│   ├── DCGAN_Model_Structure.py                # Model definitions for DCGAN
│   ├── Sound_Reconstruction_Of_Generator.py     # Audio reconstruction from generated spectrograms
│   ├── Spectrogram_Plot.py                      # Utility to plot spectrograms
│   ├── all_comparisons.py                       # Computes evaluation metrics
├── data/                # Directory for datasets (to be populated)
│   ├── Raw_Sound_Files              # Original audio files
│   ├── PreProcessed_Datasets        # Preprocessed spectrograms/MFCCs
├── results/             # Directory for generated outputs
│   ├── Loss_Datas_To_Plot            # Reconstructed audio files
├── models/            # Examples of input/output files
│   ├── backups
│   │   ├── generators
│   │   ├── discriminators
│   ├── generators
│   ├── discriminators
├── compares/            # Examples of input/output files
└── Thesis.pdf           # Full thesis document
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/YourGitHubUsername/AudioRestoration-GAN.git
   cd AudioRestoration-GAN
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Organize your dataset in the `data/` directory:
   - Place clean audio files in `data/raw/clean/`.
   - Place noisy audio files in `data/raw/noisy/`.

4. Prepare the dataset for training:
   ```bash
   python Dataloader_Prepare_And_Save.py
   ```

5. Train the GAN model:
   ```bash
   python Training_DCGAN.py
   ```

6. Reconstruct audio from a trained generator:
   ```bash
   python Sound_Reconstruction_Of_Generator.py
   ```

7. Evaluate results:
   ```bash
   python all_comparisons.py
   ```

## Evaluation Metrics

- **MSE (Mean Squared Error)**: Measures the squared error between the generated and target waveforms, highlighting accuracy.
- **RMSE (Root Mean Squared Error)**: Offers a more interpretable version of MSE by scaling back to the original units.
- **SNR (Signal-to-Noise Ratio)**: Assesses the ratio of the signal's power to the noise power in the generated output.
- **LSD (Log-Spectral Distance)**: Evaluates the spectral distortion between the original and reconstructed signals, useful for audio quality analysis.
- **STOI (Short-Time Objective Intelligibility)**: Measures the intelligibility of the reconstructed audio, essential for speech-focused tasks.

## Results

The results are presented in the `outputs/` directory. This includes:
- Reconstructed audio files (`outputs/audio`).
- Spectrogram comparisons (`outputs/spectrograms`).
- Evaluation metrics (`outputs/comparisons`).

Detailed evaluations of the experiments, along with comparisons, can be found in the thesis document. Spectrogram visualizations and waveform plots are also provided for interpretability.

## Future Work

Future iterations of this project could explore:
- Advanced architectures like Conditional GANs (cGANs) to condition audio generation on specific attributes.
- Multi-task learning for simultaneous denoising and super-resolution, improving efficiency.
- Application of diffusion models for further enhancement, as they have shown state-of-the-art performance in generative tasks.
- Integrating perceptual loss to better mimic human auditory perception.

## Acknowledgments

This project was completed as part of the Master's thesis at the **University of Padova** under the guidance of **Prof. Sergio Canazza Targon**.

Special thanks to the university's ICT department for providing computational resources and guidance throughout the development of this project.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or collaboration, please get in touch with aksoy.adnankerem@gmail.com

