#!/home/aksoyadnan/miniconda3/envs/myenv/bin/python3.12
import numpy as np
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
# Import your custom modules
import DCGAN_Model_Structure as DCGAN
import Preprocess_Afterprocess_GAN_Combined as PAGAN
import torchaudio.transforms as T

def real_imag_to_complex(data):
    """
    Convert real and imaginary parts to complex tensor.
    Args:
        data (torch.Tensor): Real and imaginary parts in separate channels.
    Returns:
        torch.Tensor: Combined complex tensor.
    """
    real_part = data[:, 0, :, :]  # Real parts
    imag_part = data[:, 1, :, :]  # Imaginary parts
    return torch.complex(real_part, imag_part)  # Combine into complex tensor


def reverse_logarithmic_scaling(log_spectrogram):
    magnitude = torch.expm1(torch.abs(log_spectrogram))  # Reverse log scaling
    phase = torch.angle(log_spectrogram)                # Preserve phase
    spectrogram = magnitude * torch.exp(1j * phase)
    return spectrogram




def mel2audio(mel_spectrogram, n_fft, n_mels, win_length, hop_length, sample_rate, logscale_bool=False):
    """
    Convert mel-spectrogram to audio waveform, with optional reverse logarithmic scaling.

    Args:
        mel_spectrogram (torch.Tensor): Input mel-spectrogram.
        n_fft (int): Number of FFT points.
        n_mels (int): Number of mel bands.
        win_length (int): Window length for Griffin-Lim.
        hop_length (int): Hop length for Griffin-Lim.
        sample_rate (int): Sampling rate of the audio.
        logscale_bool (bool): If True, reverses logarithmic scaling before conversion.

    Returns:
        torch.Tensor: Reconstructed audio waveform.
    """
    mel2spec = T.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sample_rate
    )
    # Reverse logarithmic scaling if logscale_bool is True
    if logscale_bool:
        mel_spectrogram = torch.expm1(mel_spectrogram)
        spec = mel2spec(mel_spectrogram)
        spec = spec / spec.abs().max()
    else: 
        spec = mel2spec(mel_spectrogram)
        
    # Convert mel-spectrogram back to a full spectrogram


    # Normalize the spectrogram for consistent magnitude
    

    # Convert spectrogram to waveform using Griffin-Lim
    spec2audio = T.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length
    )
    waveform = spec2audio(spec)

    return waveform




def process_and_save_audio(generator, dataloader, model_type, output_path):
    """
    Generate and save audio files from a pre-trained model.
    """
    l1 = []
    l2 = []
    for noisy, clean in dataloader:
        l1.append(noisy)
        l2.append(clean)

    with torch.no_grad():
        created = generator(l1[0])  # Process the first noisy sample

    # Convert data based on model type
    if "STFT" in model_type and "log" in model_type:
        print("####STFT LOG SCALED start to work####")
        
        # Reverse log scaling
        created = reverse_logarithmic_scaling(real_imag_to_complex(created.cpu().detach()))
        noisy = reverse_logarithmic_scaling(real_imag_to_complex(l1[0].cpu().detach()))
        real = reverse_logarithmic_scaling(real_imag_to_complex(l2[0].cpu().detach()))

        # Normalize spectrograms
        # created = created / created.abs().max()
        # noisy = noisy / noisy.abs().max()
        # real = real / real.abs().max()

        # Use consistent window parameters for iSTFT
        window = torch.hann_window(2048).to(created.device)

        created = torch.istft(created, n_fft=2048, hop_length=512, window=window, return_complex=False)
        noisy = torch.istft(noisy, n_fft=2048, hop_length=512, window=window, return_complex=False)
        real = torch.istft(real, n_fft=2048, hop_length=512, window=window, return_complex=False)

    elif "MFCC" in model_type and "log" in model_type:
        print("#### LOG SCALED MFCC Run####")
        created = mel2audio(created[0].cpu().detach(), n_fft=2048, n_mels=256, win_length=2048, hop_length=512, sample_rate=48000, logscale_bool=True)
        noisy = mel2audio(l1[0][0].cpu().detach(), n_fft=2048, n_mels=256, win_length=2048, hop_length=512, sample_rate=48000, logscale_bool=True)
        real = mel2audio(l2[0][0].cpu().detach(), n_fft=2048, n_mels=256, win_length=2048, hop_length=512, sample_rate=48000, logscale_bool=True)
    elif "MFCC" in model_type:
        print("#### MFCC Run####")
        created = mel2audio(created[0].cpu().detach(), n_fft=2048, n_mels=256, win_length=2048, hop_length=512, sample_rate=48000, logscale_bool=False)
        noisy = mel2audio(l1[0][0].cpu().detach(), n_fft=2048, n_mels=256, win_length=2048, hop_length=512, sample_rate=48000, logscale_bool=False)
        real = mel2audio(l2[0][0].cpu().detach(), n_fft=2048, n_mels=256, win_length=2048, hop_length=512, sample_rate=48000, logscale_bool=False)
    else:
        print("#### STFT Run####")
        window = torch.hann_window(2048).to(created.device)
        created = torch.istft(real_imag_to_complex(created.cpu().detach()), n_fft=2048, hop_length=512,window=window, return_complex=False)
        noisy = torch.istft(real_imag_to_complex(l1[0].cpu().detach()), n_fft=2048, hop_length=512,window=window, return_complex=False)
        real = torch.istft(real_imag_to_complex(l2[0].cpu().detach()), n_fft=2048, hop_length=512,window=window, return_complex=False)

    # Flatten or reshape data for saving
    created = created.squeeze().numpy()
    noisy = noisy.squeeze().numpy()
    real = real.squeeze().numpy()

    # Normalize audio
    created = created / np.abs(created).max()
    noisy = noisy / np.abs(noisy).max()
    real = real / np.abs(real).max()

    # Debug shapes and ranges
    print(f"Created shape: {created.shape}, range: [{created.min()}, {created.max()}]")
    print(f"Noisy shape: {noisy.shape}, range: [{noisy.min()}, {noisy.max()}]")
    print(f"Real shape: {real.shape}, range: [{real.min()}, {real.max()}]")

    # Save the generated, noisy, and real audio
    sf.write(f"{output_path}/Generated.wav", created.astype(np.float32), 48000)
    sf.write(f"{output_path}/Noisy.wav", noisy.astype(np.float32), 48000)
    sf.write(f"{output_path}/Real.wav", real.astype(np.float32), 48000)

    print("Audio files saved successfully!")






def main():
    # Load the pre-trained generator
    model_type = "STFT"  # Choose between "MFCC", "STFT", "MFCC_log", "STFT_log".
    

    output_path = "Outputs"  # Specify your output folder

    
    # Load the dataset
    # dataset = PAGAN.AudioDataset(
    #     clean_files=["C:/Users/adnan/PD_ICT_Master_Thesis/16bit_48kHz/Original_Sample_Full.wav"],
    #     noisy_files_list=['C:/Users/adnan/PD_ICT_Master_Thesis/16bit_48kHz/LP_150Hz_24dBoct_WhiteNoise_30RMS_Gaussian.wav'],
    #     start_time=2,
    #     duration=6,
    #     segment_length=0.5
    # )

    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    dataloader = torch.load(
        "stft_updated"
    )   

    # dataloader = torch.load(
    #     "Deneme_mfcclog"
    # )

    data_batch, _ = next(iter(dataloader))  # Get one batch
    _, channel_count, freq_bins, frame_rate = data_batch.shape  # Extract time frames

    generator = DCGAN.Generator(channel_count=channel_count,freq_bins=freq_bins,frame_rate=frame_rate)
    generator.load_state_dict(torch.load("GithupPreparationGenerator"))
    generator.eval()
    generator.cpu()


    # Output path for audio files
    output_path = "Outputs"

    # Process and save audio files
    process_and_save_audio(generator, dataloader, model_type, output_path)


if __name__ == "__main__":
    main()
