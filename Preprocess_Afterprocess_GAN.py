import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader


def preprocess_audio(audio_path, start_time, duration, segment_length):
    waveform, sr = torchaudio.load(audio_path)
    
    # Ensure waveform is mono if it has more than one channel
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Trim the waveform to the desired 30-second section, Trimming is adjustable
    # by changing start_time and duration, you can change the trimming.
    end_sample = int(start_time * sr) + int(duration * sr)
    start_sample = int(start_time * sr)
    trimmed_waveform = waveform[:, start_sample:end_sample]

    # Calculate the number of samples in each segment
    num_samples = int(segment_length * sr)
    total_samples = trimmed_waveform.shape[1]
    segments = []

    # Break down the trimmed waveform into segments, "segment_lengths= 2" means that it will divide into 2 second segments
    for start in range(0, total_samples, num_samples):
        if start + num_samples <= total_samples:
            segment = trimmed_waveform[:, start:start + num_samples]
            segments.append(segment)

    segment_spectrograms = []

    # Generate a spectrogram for each segment
    for segment in segments:
        window = torch.hann_window(2048)  # Ensure the window is on the same device
        spectrogram = torch.stft(segment, n_fft=2048, hop_length=512, window=window, return_complex=True)
        segment_spectrograms.append(spectrogram)
    
    return segment_spectrograms, sr


def spectrogram_to_audio(spectrogram): # turns the spectrogram to waveform format to reconstruct the audio 
    # Ensure window is on the same device as spectrogram
    window = torch.hann_window(2048, device=spectrogram.device)
    # Assuming 'spectrogram' is a complex tensor from the decoder
    waveform = torch.istft(spectrogram, n_fft=2048, hop_length=512, window=window,return_complex=False)
    return waveform.squeeze().numpy()


def complex_to_real_imag(data):
    # Split complex data into real and imaginary parts
    real_part = data.real
    imag_part = data.imag
    # Stack the real and imaginary parts along the channel dimension
    c_r = torch.cat([real_part, imag_part], dim=0)
    return c_r  # Concatenate along the channel axis



class AudioDataset(Dataset):
    def __init__(self, clean_files, noisy_files_list, start_time, duration, segment_length):
        self.clean_data = []
        self.noisy_data = []
        
        # Preprocess all clean files once and reuse
        for clean_file in clean_files:
            clean_spectrograms, _ = preprocess_audio(clean_file, start_time, duration, segment_length)
            for spec in clean_spectrograms:
                # print(spec.shape)
                # print(complex_to_real_imag(spec).shape)
                self.clean_data.append(complex_to_real_imag(spec))
        
        # Preprocess all noisy files
        for noisy_file in noisy_files_list:
            noisy_spectrograms, _ = preprocess_audio(noisy_file, start_time, duration, segment_length)
            for spec in noisy_spectrograms:
                self.noisy_data.append(complex_to_real_imag(spec))

    def __len__(self):
        # Return the number of noisy samples, which are more frequent
        return len(self.noisy_data)

    def __getitem__(self, idx):
        # Circular access for clean data if there are fewer clean than noisy samples
        clean_idx = idx % len(self.clean_data)
        noisy_spectrogram = self.noisy_data[idx]
        clean_spectrogram = self.clean_data[clean_idx]
        return noisy_spectrogram, clean_spectrogram
    
# Initialize dataset and dataloader
# dataset = AudioDataset(clean_files=['path_to_clean_file.wav'], noisy_files_list=['noisy1.wav', 'noisy2.wav', ...])
# dataloader = DataLoader(dataset, batch_size=10, shuffle=True)









