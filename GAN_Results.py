import numpy as np
import torch
import torch.nn as nn
import DCGAN 
import Preprocess_Afterprocess_GAN as PAGAN 
import torch.optim as optim
from tqdm import tqdm
import soundfile as sf


# Initialize dataset and dataloader
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

# dataset = PAGAN.AudioDataset(clean_files=['16bit_48kHz\Original_Sample_Full.wav'], 
#                              noisy_files_list=['Missing_frequencies\M1.wav'],
#                                                start_time=0,
#                                                duration=30,
#                                                segment_length=2)





dataloader = PAGAN.DataLoader(dataset, batch_size=1, shuffle=True)
#(self, clean_files, noisy_files_list, start_time, duration, segment_length)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = DCGAN.Generator()


generator.load_state_dict(torch.load("GAN_file\DCGAN\Generator_trained_420step_v3_lr0002_dropout0_BCEloss_BetterD_Leaky_NOsmoothlabels"  )) #DC GAN
# generator.load_state_dict(torch.load("GAN_file\Generator_trained_400_lr0002_dropout025_MSExBCEloss" ))




l1 = []
l2 = []
for noisy,clean in dataloader:
    l1.append(noisy)
    l2.append(clean)

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

# print(l1[8])
created = generator(l1[47])



created_g= real_imag_to_complex(created.detach())
original = real_imag_to_complex(l2[47])
noisy = real_imag_to_complex(l1[47])
# print(printy)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
created_g_audio = PAGAN.spectrogram_to_audio(created_g[0])
original_audio = PAGAN.spectrogram_to_audio(original[0])
noisy_audio = PAGAN.spectrogram_to_audio(noisy[0])
print(created_g_audio)
print(original_audio)

sf.write('GAN_file\DCGAN\Generator_trained_500_v3_lr0002_dropout0_BCEloss_BetterD_Leaky_NOsmoothlabels_recon47_DCGAN.wav', created_g_audio, samplerate=48000)
sf.write('GAN_file\DCGAN\Generator_trained_500_v3_lr0002_dropout0_BCEloss_BetterD_Leaky_NOsmoothlabels_noise47_DCGAN.wav', noisy_audio, samplerate=48000)
sf.write('GAN_file\DCGAN\Generator_trained_500_v3_lr0002_dropout0_BCEloss_BetterD_Leaky_NOsmoothlabels_original47_DCGAN.wav', original_audio, samplerate=48000)