import torch
import torchaudio
import matplotlib.pyplot as plt
import librosa

def plot_specgram(waveform: torch.Tensor, path: str = None, title: str = None, ylabel: str = "Frequency Bin", ax=None):
    """
    Plots a spectrogram of the given waveform and optionally saves it to a specified path.

    Args:
        waveform (torch.Tensor): Input waveform tensor.
        path (str, optional): Path to save the spectrogram image. If None, the plot will be displayed.
        title (str): Title of the spectrogram plot.
        ylabel (str): Label for the y-axis.
        ax (matplotlib.axes.Axes, optional): Axes object for plotting.
    """
    # Create a spectrogram
    transform = torchaudio.transforms.Spectrogram(n_fft=800)
    spectrogram = transform(waveform).squeeze(0)  # Remove channel dimension if present

    # Apply logarithmic scaling to prevent zeros
    log_spectrogram = torch.log1p(spectrogram)

    # Create the plot
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))
    if title:
        ax.set_title(title)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time Frame")
    ax.imshow(librosa.power_to_db(log_spectrogram.numpy()), origin="lower", aspect="auto", interpolation="nearest")

    # Save or display the plot
    if path:
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    
    # Load example waveforms
    waveform_original, _ = torchaudio.load("LoadPath/filename_original.wav")
    waveform_noisy, _ = torchaudio.load("LoadPath/filename_noisy.wav")
    waveform_reconstructed, _ = torchaudio.load("LoadPath/filename_reconstruction.wav")

    # Plot spectrograms
    plot_specgram(waveform_original, path="original_spectrogram.png", title="Original Audio")
    plot_specgram(waveform_noisy, path="noisy_spectrogram.png", title="Noisy Audio")
    plot_specgram(waveform_reconstructed, path="reconstructed_spectrogram.png", title="Reconstructed Audio")

