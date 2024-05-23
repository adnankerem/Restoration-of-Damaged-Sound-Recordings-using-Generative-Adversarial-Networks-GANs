import torch
import torchaudio
import Preprocess_Afterprocess_GAN as PAGAN
import soundfile as sf
import numpy as np

# def preprocess_audio(audio_path):
#     waveform, sr = torchaudio.load(audio_path)
#     # Ensure single channel (mono)
#     if waveform.shape[0] > 1:
#         waveform = torch.mean(waveform, dim=0, keepdim=True)
#     # Create Hann window on the same device as waveform
#     window = torch.hann_window(2048, device=waveform.device)
#     # Compute the complex spectrogram
#     spectrogram = torch.stft(waveform, n_fft=2048, hop_length=512, window=window, return_complex=True)
#     return spectrogram, sr

# spectrogram ,_= preprocess_audio("16bit_48kHz\LP_1000Hz_24dBoct_WhiteNoise_30RMS_Gaussian.wav")
# # print(spectrogram.shape)

# # Initialize dataset and dataloader


# clean_waveform, sr = PAGAN.preprocess_audio("16bit_48kHz\LP_1000Hz_24dBoct_WhiteNoise_30RMS_Gaussian.wav", start_time=0, duration=30,segment_length=2)

# print(clean_waveform[0].shape)
# reconstructed_audio = PAGAN.spectrogram_to_audio(clean_waveform[0])
# print(reconstructed_audio.shape)
# sf.write('reconstructed_audio.wav', reconstructed_audio, samplerate=sr)  # Ensure 'sr' is defined as the sample rate



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# spectrogram, sr = preprocess_audio("16bit_48kHz\Original_Sample_Full.wav",0,30,2)
# if spectrogram is not None:
#     reconstructed_audio = spectrogram_to_audio(spectrogram[3])
#     print("Audio reconstructed successfully.")
#     # Code to play or save `reconstructed_audio` here
# else:
#     print("Failed to preprocess audio.")

# # Save the output
# import soundfile as sf
# sf.write('reconstructed_audio1original_segmented.wav', reconstructed_audio, samplerate=48000)


import Preprocess_Afterprocess_GAN as PAGAN 
dataset = PAGAN.AudioDataset(clean_files=['16bit_48kHz\Original_Sample_Full.wav'], 
                             noisy_files_list=['16bit_48kHz\LP_150Hz_24dBoct_WhiteNoise_30RMS_Gaussian.wav', 
                                               '16bit_48kHz\LP_250Hz_24dBoct_WhiteNoise_30RMS_Gaussian.wav', 
                                               '16bit_48kHz\LP_500Hz_24dBoct_WhiteNoise_30RMS_Gaussian.wav',
                                               '16bit_48kHz\LP_1000Hz_24dBoct_WhiteNoise_30RMS_Gaussian.wav',
                                               '16bit_48kHz\LP_1500Hz_24dBoct_WhiteNoise_30RMS_Gaussian.wav',
                                               '16bit_48kHz\LP_2000Hz_24dBoct_WhiteNoise_30RMS_Gaussian.wav',
                                               '16bit_48kHz\LP_3000Hz_24dBoct_WhiteNoise_30RMS_Gaussian.wav'],
                                               start_time=0,
                                               duration=30,
                                               segment_length=2)
dataloader = PAGAN.DataLoader(dataset, batch_size=1, shuffle=True)
l1 = []
l2 = []

for noisy,clean in dataloader:
    l1.append(noisy)
    l2.append(clean)

print(len(l1))
print(len(l2))

# real_part = l2[18].real
# imag_part = l2[18].imag
# c_r = torch.cat([real_part, imag_part], dim=1)
# print(c_r)

# print(c_r.shape)

print(l2[18])

print(l2[18].shape)
def real_imag_to_complex(data):
    # Assume data has shape [batch_size, 2, frequency, time]
    # Split the real and imaginary parts which are concatenated along the channel dimension
    real_part = data[:, 0, :, :]  # Real parts
    imag_part = data[:, 1, :, :]  # Imaginary parts
    
    # Stack them along a new dimension to make them interleaved in the last dimension
    combined = torch.stack((real_part, imag_part), dim=-1)  # Shape [batch_size, frequency, time, 2]
    
    # Convert to complex
    complex_data = torch.view_as_complex(combined)
    return complex_data

printy = real_imag_to_complex(l2[18])
print(printy.shape)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
aaa = PAGAN.spectrogram_to_audio(printy)
import soundfile as sf
sf.write('reconstructed_clean_audio18_recomplexed.wav', aaa, samplerate=48000)
# if l1 is not None:
#     noisy_reconstructed_audio_c = torch.stack(l1[18], dim=-1)
#     noisy_reconstructed_audio = PAGAN.spectrogram_to_audio(noisy_reconstructed_audio_c)

#     clean_reconstructed_audio_c = torch.stack(l2[18], dim=-1)
#     clean_reconstructed_audio = PAGAN.spectrogram_to_audio(clean_reconstructed_audio_c)
    
#     print("Audio reconstructed successfully.")
#     # Code to play or save `reconstructed_audio` here
# else:
#     print("Failed to preprocess audio.")

# # Save the output
# import soundfile as sf
# sf.write('reconstructed_clean_audio18.wav', clean_reconstructed_audio, samplerate=48000)
# sf.write('reconstructed_noisy_audio18_c.wav', noisy_reconstructed_audio, samplerate=48000)