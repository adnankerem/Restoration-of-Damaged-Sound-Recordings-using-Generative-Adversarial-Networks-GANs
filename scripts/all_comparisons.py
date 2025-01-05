import os
import torchaudio
import numpy as np
from pystoi import stoi
import soundfile as sf
from scipy import signal
import librosa

# Function to compute SNR
def compute_snr(original, reconstructed):
    noise = original - reconstructed
    signal_power = np.sum(original ** 2)
    noise_power = np.sum(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Function to compute LSD
def compute_lsd(original, reconstructed, n_fft=2048):
    original_spec = np.abs(librosa.stft(original, n_fft=n_fft))
    reconstructed_spec = np.abs(librosa.stft(reconstructed, n_fft=n_fft))
    log_diff = 20 * (np.log10(original_spec + 1e-8) - np.log10(reconstructed_spec + 1e-8))
    lsd = np.mean(np.sqrt(np.mean(log_diff ** 2, axis=0)))
    return lsd

# Function to compute MSE and RMSE
def compute_mse_rmse(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    rmse = np.sqrt(mse)
    return mse, rmse

# Function to compute STOI
def compute_stoi_value(original, reconstructed, sr):
    return stoi(original, reconstructed, sr, extended=False)

# Main function to process specific lists of files and append results to a single output file
def process_comparisons(comparisons, output_file):
    results = []
    
    for comparison in comparisons:
        if len(comparison) != 4:
            continue  # Ensure each comparison has exactly 4 items (name, original, generated, noisy)

        experiment_name, original_file, generated_file, noisy_file = comparison

        original, sr = torchaudio.load(original_file)
        generated, _ = torchaudio.load(generated_file)

        original = original.squeeze().numpy()
        generated = generated.squeeze().numpy()

        # Compute metrics
        mse, rmse = compute_mse_rmse(original, generated)
        snr = compute_snr(original, generated)
        lsd = compute_lsd(original, generated)
        stoi_value = compute_stoi_value(original, generated, sr)

        results.append({
            'Experiment': experiment_name,
            'MSE': mse,
            'RMSE': rmse,
            'SNR': snr,
            'LSD': lsd,
            'STOI': stoi_value
        })
    
    # Save results to a single text file
    with open(output_file, 'a') as f:  # 'a' for appending results to the file
        for result in results:
            f.write(f"Experiment: {result['Experiment']}\n")
            f.write(f"MSE: {result['MSE']:.4f}\n")
            f.write(f"RMSE: {result['RMSE']:.4f}\n")
            f.write(f"SNR: {result['SNR']:.4f}\n")
            f.write(f"LSD: {result['LSD']:.4f}\n")
            f.write(f"STOI: {result['STOI']:.4f}\n")
            f.write("\n")

# Example of how to structure your file comparisons


comparisons = [
    # Each list contains [experiment_name, original_file, generated_file, noisy_file]
    ["count or name of experiment", "original sound file path.wav", "reconstructed sound file path.wav", "noisy or damaged sound file path.wav"],

    # Add more comparisons here
]
# Output file path
output_file = "../compares/compare outputs.txt"

# Run the processing function
process_comparisons(comparisons, output_file)
