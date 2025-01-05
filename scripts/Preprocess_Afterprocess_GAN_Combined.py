#!/home/aksoyadnan/miniconda3/envs/myenv/bin/python3.12
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchaudio.transforms as T


def preprocess_audio(
    audio_path: str,
    start_time: int,
    duration: int,
    segment_length: int,
    mfcc_bool: str,
    stft_bool: str,
    logscale_bool: str,
):
    waveform, sr = torchaudio.load(audio_path)
    if mfcc_bool == stft_bool or (not mfcc_bool and not stft_bool):
        raise ValueError(
            "Invalid configuration: Set either 'mfcc_bool' or 'stft_bool' to True, but not both."
        )

    else:
        # Ensure waveform is mono if it has more than one channel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Trim the waveform to the desired section
        end_sample = int(start_time * sr) + int(duration * sr)
        start_sample = int(start_time * sr)
        trimmed_waveform = waveform[:, start_sample:end_sample]

        # Calculate the number of samples in each segment
        num_samples = int(segment_length * sr)
        total_samples = trimmed_waveform.shape[1]
        segments = []

        # Break down the trimmed waveform into segments
        for start in range(0, total_samples, num_samples):
            if start + num_samples <= total_samples:
                segment = trimmed_waveform[:, start : start + num_samples]
                segments.append(segment)

        if mfcc_bool == True:
            segment_mfccs = []

            # Generate MFCC (with or without log scaling, controlled by logscale_bool) for each segment
            for segment in segments:
                n_fft = 2048
                mfcc = audio2mel_scale(
                    waveform=segment,
                    sample_rate=sr,
                    n_fft=n_fft,
                    hop_length=n_fft // 4,
                    n_mels=256,
                    logscale_bool=logscale_bool,  # Pass the log scaling choice here
                )
                mfcc_data = mfcc_data.unsqueeze(1)
                segment_mfccs.append(mfcc)

            return segment_mfccs, sr

        elif stft_bool:
            segment_spectrograms = []
            # Generate STFT data
            for segment in segments:
                window = torch.hann_window(2048)
                spectrogram = torch.stft(
                    segment,
                    n_fft=2048,
                    hop_length=512,
                    window=window,
                    return_complex=True,
                )

                # Convert complex to real
                spectrogram_real = complex_to_real_imag(spectrogram)

                # Logarithmic scaling
                if logscale_bool:
                    magnitude = torch.log1p(torch.abs(spectrogram_real))
                    spectrogram_real = magnitude  # Replace with log-scaled magnitude

                segment_spectrograms.append(spectrogram_real)
            return segment_spectrograms, sr


def mel2audio(
    mel_spectrogram: torch.Tensor,
    n_fft: int,
    n_mels: int,
    win_length: int,
    hop_length: int,
    sample_rate: int,
    logscale_bool: bool = False,
) -> torch.Tensor:
    """
    Converts a mel-spectrogram back to audio, reversing logarithmic scaling if applied.

    Args:
        mel_spectrogram (torch.Tensor): Input mel-spectrogram.
        n_fft (int): Number of FFT points.
        n_mels (int): Number of mel bands.
        win_length (int): Window length for Griffin-Lim.
        hop_length (int): Hop length for Griffin-Lim.
        sample_rate (int): Sample rate of the audio.
        logscale_bool (bool): If True, reverses the logarithmic scaling.

    Returns:
        torch.Tensor: Reconstructed audio waveform.
    """
    if logscale_bool:
        # Reverse logarithmic scaling: exp(mel_spectrogram) - 1
        mel_spectrogram = torch.expm1(mel_spectrogram)

    mel2spec = T.InverseMelScale(
        n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sample_rate
    ).to(mel_spectrogram.device)
    spec2audio = T.GriffinLim(
        n_fft=n_fft, win_length=win_length, hop_length=hop_length
    ).to(mel_spectrogram.device)

    # Convert mel-spectrogram to spectrogram
    spec = mel2spec(mel_spectrogram)

    # Convert spectrogram to waveform
    audio = spec2audio(spec)

    return audio


def audio2mel_scale(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    logscale_bool: bool = False,
) -> torch.Tensor:
    """
    Converts waveform to a mel spectrogram with optional logarithmic scaling.

    Args:
        waveform (torch.Tensor): Input audio waveform.
        sample_rate (int): Sample rate of the audio.
        n_fft (int): Number of FFT points.
        hop_length (int): Hop length for the spectrogram.
        n_mels (int): Number of mel bands.
        logscale_bool (bool): If True, applies logarithmic scaling.

    Returns:
        torch.Tensor: Mel spectrogram (log-scaled if specified).
    """
    mel_scale = T.MelScale(
        n_mels=n_mels, sample_rate=sample_rate, n_stft=n_fft // 2 + 1
    )
    spectrogram_transform = T.Spectrogram(
        n_fft=n_fft, win_length=None, hop_length=hop_length
    )

    # Generate spectrogram
    spectrogram = spectrogram_transform(waveform)

    # Convert to mel scale
    mel = mel_scale(spectrogram)

    # Apply logarithmic scaling if specified
    if logscale_bool:
        mel = torch.log1p(mel)

    return mel


def spectrogram_to_audio(spectrogram, logscale_bool=False):
    """
    Converts a spectrogram to waveform format and optionally reverses the logarithmic scaling.

    Args:
        spectrogram (torch.Tensor): Input spectrogram (complex tensor).
        logscale_bool (bool): If True, reverses the logarithmic scaling before conversion.

    Returns:
        numpy.ndarray: Reconstructed audio waveform.
    """
    if logscale_bool:
        # Reverse logarithmic scaling: exp(spectrogram) - 1
        magnitude = torch.abs(spectrogram)  # Extract magnitude
        phase = torch.angle(spectrogram)  # Extract phase
        reversed_magnitude = torch.expm1(
            magnitude
        )  # Apply exponential scaling (inverse of log1p)

        # Reconstruct spectrogram with original phase
        spectrogram = reversed_magnitude * torch.exp(1j * phase)

    # Ensure window is on the same device as spectrogram
    window = torch.hann_window(2048, device=spectrogram.device)

    # Convert spectrogram to waveform
    waveform = torch.istft(
        spectrogram, n_fft=2048, hop_length=512, window=window, return_complex=False
    )

    return waveform.squeeze().numpy()


def complex_to_real_imag(data):
    # Split complex data into real and imaginary parts
    real_part = data.real
    imag_part = data.imag
    # Stack the real and imaginary parts along the channel dimension
    c_r = torch.cat([real_part, imag_part], dim=0)
    return c_r  # Concatenate along the channel axis


class AudioDataset(Dataset):
    def __init__(
        self,
        clean_files: list,
        noisy_files_list: list,
        start_time: int,
        duration: int,
        segment_length: int,
        mfcc_bool: bool,
        stft_bool: bool,
        logscale_bool: bool,
    ):
        self.clean_data = []
        self.noisy_data = []

        # Preprocess all clean files once and reuse
        for clean_file in clean_files:
            clean_mfccs, _ = preprocess_audio(
                clean_file,
                start_time,
                duration,
                segment_length,
                mfcc_bool,
                stft_bool,
                logscale_bool,
            )
            self.clean_data.extend(clean_mfccs)

        # Preprocess all noisy files
        for noisy_file in noisy_files_list:
            noisy_mfccs, _ = preprocess_audio(
                noisy_file,
                start_time,
                duration,
                segment_length,
                mfcc_bool,
                stft_bool,
                logscale_bool,
            )
            self.noisy_data.extend(noisy_mfccs)

    def __len__(self):
        return len(self.noisy_data)

    def __getitem__(self, idx):
        clean_idx = idx % len(self.clean_data)
        noisy_mfcc = self.noisy_data[idx]
        clean_mfcc = self.clean_data[clean_idx]
        return noisy_mfcc, clean_mfcc


# Example Usage
# dataset = AudioDataset(clean_files=['path_to_clean_file.wav'], noisy_files_list=['path_to_noisy_file1.wav', 'path_to_noisy_file2.wav'], start_time=0, duration=30, segment_length=2)
# dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Example plot
# waveform, sr = torchaudio.load('path_to_audio_file.wav')
# mfcc = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=13)(waveform)
# plot_mfcc(mfcc, epoch=1, name="example")

# Example inversion
# reconstructed_waveform = mfcc_to_audio(mfcc[0], sr)
# sf.write('reconstructed_audio.wav', reconstructed_waveform, sr)
