import numpy as np
import torch
import torch.nn as nn
from GAN import Generator,Discriminator,init_weights
import Preprocess_Afterprocess_GAN as PAGAN 
import torch.optim as optim
from tqdm import tqdm



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
dataloader = PAGAN.DataLoader(dataset, batch_size=4, shuffle=True)
#(self, clean_files, noisy_files_list, start_time, duration, segment_length)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = Generator()

mse_loss = nn.MSELoss()

def train(generator, dataloader, device, num_epochs=50):
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    generator.to(device)

    for epoch in range(num_epochs):
        for noisy_spectrograms, clean_spectrograms in dataloader:
            noisy_data = noisy_spectrograms.to(device)
            real_data = clean_spectrograms.to(device)
            batch_size = real_data.size(0)

            # Train Generator
            generator.zero_grad()
            fake_data = generator(noisy_data)
            # Use MSE Loss for generator
            g_loss = mse_loss(fake_data, real_data)  # Compare directly with clean spectrograms
            print(g_loss)
            g_loss.backward()
            optimizer_g.step()
        if (epoch % 1 == 0):
            print(fake_data)
        # Logging for insight, not necessary for training
        if (epoch % 1 == 0):  # Log every 10 epochs
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                    f' G_loss: {g_loss.item():.4f}')
    print(fake_data)
    # Optionally save models
    

# Example setup for running the training
generator.load_state_dict(torch.load("GAN_file\Generator_trained_5_lr0002_MSEloss_withoutDiscri"))

# dataloader = DataLoader(dataset, batch_size=10, shuffle=True)  # Assuming 'dataset' is properly defined
# generator.apply(init_weights)
# Call train function
train(generator=generator, dataloader=dataloader, device=device, num_epochs=5)




torch.save(generator.state_dict(),"GAN_file\Generator_trained_10_lr0002_MSEloss_withoutDiscri" )
