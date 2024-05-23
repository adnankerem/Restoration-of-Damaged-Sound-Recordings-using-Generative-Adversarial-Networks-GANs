import numpy as np
import torch
import torch.nn as nn
from DCGAN import Generator,Discriminator,init_weights
import Preprocess_Afterprocess_GAN as PAGAN 
import torch.optim as optim
from tqdm import tqdm
import pickle



# Initialize dataset and dataloader
dataset = PAGAN.AudioDataset(clean_files=['16bit_48kHz\Original_Sample_Full.wav'], 
                             noisy_files_list=['16bit_48kHz\LP_150Hz_24dBoct_WhiteNoise_30RMS_Gaussian.wav', 
                                               '16bit_48kHz\LP_250Hz_24dBoct_WhiteNoise_30RMS_Gaussian.wav', 
                                               '16bit_48kHz\LP_500Hz_24dBoct_WhiteNoise_30RMS_Gaussian.wav',
                                               '16bit_48kHz\LP_1000Hz_24dBoct_WhiteNoise_30RMS_Gaussian.wav',
                                               '16bit_48kHz\LP_1500Hz_24dBoct_WhiteNoise_30RMS_Gaussian.wav',
                                               '16bit_48kHz\LP_2000Hz_24dBoct_WhiteNoise_30RMS_Gaussian.wav',
                                               '16bit_48kHz\LP_3000Hz_24dBoct_WhiteNoise_30RMS_Gaussian.wav'],
                                               start_time=2,
                                               duration=62,
                                               segment_length=2)
dataloader = PAGAN.DataLoader(dataset, batch_size=5, shuffle=True)
#(self, clean_files, noisy_files_list, start_time, duration, segment_length)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = Generator()
discriminator = Discriminator()

# generator.apply(init_weights)  
# discriminator.apply(init_weights)

g_loss_list = []
d_loss_list = []

def train(generator, discriminator, dataloader, device, num_epochs=50):
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion_bce = nn.BCEWithLogitsLoss()
    global g_loss_list,d_loss_list
    generator.to(device)
    discriminator.to(device)

    for epoch in range(num_epochs):
        for noisy_spectrograms, clean_spectrograms in dataloader:
            noisy_data = noisy_spectrograms.to(device)
            real_data = clean_spectrograms.to(device)
            batch_size = real_data.size(0)

            # Train Discriminator
            discriminator.zero_grad()
            real_output = discriminator(real_data)
            real_labels = torch.full((batch_size, 1), 0.9, device=device)  # Label smoothing
            fake_labels = torch.full((batch_size, 1), 0.1, device=device)
            d_loss_real = criterion_bce(real_output, real_labels)

            fake_data = generator(noisy_data)
            fake_output = discriminator(fake_data.detach())
            d_loss_fake = criterion_bce(fake_output, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            print("d_loss_real =",d_loss_real)
            print("d_loss_fake =",d_loss_fake)
            print("d_loss =",d_loss)
            d_loss_list.append(d_loss)
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            for i in range(3):
                generator.zero_grad()
                # Adversarial loss (BCE)
                if i == 0:
                    trick_output = discriminator(fake_data)
                    print(trick_output)
                else:
                    fake_data = generator(noisy_data)
                    trick_output = discriminator(fake_data)

                g_loss_adv = criterion_bce(trick_output, real_labels)
                print("g_loss =",g_loss_adv)
                # Content loss (MSE)
                # g_loss_mse = criterion_mse(fake_data, real_data)
                # print(g_loss_mse)
                # Combine losses
                g_loss = g_loss_adv 
                if i == 2:
                    g_loss_list.append(g_loss)
                g_loss.backward()
                optimizer_g.step()

        if (epoch % 1 == 0):
            print(fake_data)
        # Logging for insight, not necessary for training
        if (epoch % 1 == 0):  # Log every 10 epochs
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                    f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')
        if (epoch % 5 == 0):
            torch.save(generator.state_dict(),"GAN_file\DCGAN\Generator_trained_40step_v3_lr0002_dropout0_BCEloss_BetterG_Leaky" )
            torch.save(discriminator.state_dict(),"GAN_file\DCGAN\Discriminator_trained_40step_v3_lr0002_dropout0_BCEloss_BetterG_Leaky")
    print(fake_data)
    # Optionally save models
    

# # Example setup for running the training
# generator.load_state_dict(torch.load("GAN_file\DCGAN\Generator_trained_40_v3_lr0002_dropout0_BCEloss_BetterG_Leaky" ))
# discriminator.load_state_dict(torch.load("GAN_file\DCGAN\Discriminator_trained_40_v3_lr0002_dropout0_BCEloss_BetterG_Leaky"))

# dataloader = DataLoader(dataset, batch_size=10, shuffle=True)  # Assuming 'dataset' is properly defined

# Call train function
train(generator, discriminator, dataloader, device, num_epochs=1)

with open("g_loss_list_GLeaky_40to50", "wb") as fp:   #Pickling
      pickle.dump(g_loss_list, fp)
with open("d_loss_list_GLeaky_40to50", "wb") as fp:   #Pickling
      pickle.dump(d_loss_list, fp)


torch.save(generator.state_dict(),"GAN_file\DCGAN\deneme" )
torch.save(discriminator.state_dict(),"GAN_file\DCGAN\deneme")
