#!/home/aksoyadnan/miniconda3/envs/myenv/bin/python3.12
import numpy as np
import torch
import torch.nn as nn
from DCGAN_Model_Structure import Generator, Discriminator, init_weights
import Preprocess_Afterprocess_GAN_Combined as PAGAN
import torch.optim as optim
import pickle


'''
    ###WARNING### If you don't have a ready dataset, you can upload the clean and noisy vaw
    files here and prepare the dataloader With the ""Preprocess_Afterprocess_GAN_Combined"""
    library you can turn the vaw files into divided segments and turn these divided
    segments into logarithmically scaled MFCC datas
    With the "Start_Time" variable you can choose the start time to take the segments (if you
    have any parts to discard at the start), with "duration" variable you can choose the end time
    (if you have any parts to discard at the end), with "segment_lenght" variable, you can
    pick the segment lenghts in seconds. According to the "mfcc_bool,stft_bool,logscale_bool" values
    the function will prepare the data.
'''


# dataloader = torch.load(
#     "Deneme_mfcclog"
# )

dataloader = torch.load("Deneme_stftlog")

# Get the frame rate from the dataloader
data_batch, _ = next(iter(dataloader))  # Get one batch
_, channel_count, freq_bins, frame_rate = data_batch.shape  # Extract time frames
print("Data Batch Shape: ", data_batch.shape)
print(f"Detected frame rate: {frame_rate}")
print(f"Detected channel count: {channel_count}")
print(f"Detected values: {freq_bins}")


#### With the Pre-Process-After-Process library you can turn the vaw files into divided segments to use in the code

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = Generator(
    frame_rate=frame_rate, channel_count=channel_count, freq_bins=freq_bins
)
discriminator = Discriminator(
    frame_rate=frame_rate, channel_count=channel_count, freq_bins=freq_bins
)

# generator.apply(init_weights)
# discriminator.apply(init_weights)

g_loss_list = []
d_loss_list = []


def train(
    generator,
    discriminator,
    dataloader,
    device,
    num_epochs=50,
    Generator_Save_Path=None,
    Discriminator_Save_Path=None,
):
    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=0.0008, betas=(0.5, 0.999)
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=0.0008, betas=(0.5, 0.999)
    )
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()
    global g_loss_list, d_loss_list
    generator.to(device)
    discriminator.to(device)
    discri_train_counter = 0
    for epoch in range(num_epochs):
        for noisy_spectrograms, clean_spectrograms in dataloader:
            noisy_data = noisy_spectrograms.to(device)
            real_data = clean_spectrograms.to(device)
            batch_size = real_data.size(0)

            # Train Discriminator
            discriminator.zero_grad()
            real_output = discriminator(real_data)

            #### OPTIONAL ####
            ### In this case, to prevent Discriminator to Overpower the Generator, we have used Label Smoothing
            ### In our case, we do not have a uniform data, therefore it is way more harder for our
            ### generator to learn from the Discriminator
            ###### The Label Smoothing ranges we have used in this application is
            ###### on the real_labels: from 0.9 to 0.9995
            ###### on the fake_labels: from 0.1 to 0.0005
            # real_labels = torch.full((batch_size, 1), 0.9995, device=device)  # Label smoothing
            # fake_labels = torch.full((batch_size, 1), 0.0005, device=device)

            ###### OPTIONAL ###### : This is the labels without the Label Smoothing
            real_labels = torch.ones(
                batch_size, 1, device=device
            )  # Set to 1 for real images
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Scoring is made with Binary Cross Entropy
            d_loss_real = criterion_bce(real_output, real_labels)
            ### Mean Square Error is not useful for us in this case because of the aim of the thesis
            ### Using Mean Square Error cause us to overfit the used data, and that can cause our model to
            ### not work properly if we try it another sound file which we did not included in the dataset

            # Creating fake data
            fake_data = generator(noisy_data)

            # scoring the fake data we have used
            fake_output = discriminator(fake_data.detach())

            # Scoring is made with Binary Cross Entropy
            d_loss_fake = criterion_bce(fake_output, fake_labels)
            # Summed up losses
            d_loss = d_loss_real + d_loss_fake

            if d_loss_fake < 0.1:
                """
                If the loss discriminator loss is too low, and if its gets lower or if its
                prevent the Generator to learn propperly, after some training steps of Discriminator,
                we are stopping the Discriminator training until the Generator can learn a bit more and
                to increase the loss of the Discriminator above the treshold. If this happens the Discriminator
                can proceed the training.
                """

                discri_train_counter = discri_train_counter + 1
                if discri_train_counter > 1:
                    #### If the discriminator loss is lower than tresh hold "X" times, the Discriminator training will be stopped
                    print("Discriminator will not train")
                    if discri_train_counter > 15:
                        #### If the training of the Discriminator stopped for more than "X" iterations
                        #### Discriminator will train atleast once
                        d_loss_list.append(d_loss)
                        d_loss.backward()
                        optimizer_d.step()
                        discri_train_counter = 0
                else:
                    #### If the discriminator loss got higher than the threshold, Discriminator
                    #### will train at that iteration
                    d_loss_list.append(d_loss)
                    d_loss.backward()
                    optimizer_d.step()
                    print(
                        "discriminator trained: "
                        + str(discri_train_counter)
                        + " <= Discriminator will stop if this value reaches X "
                    )

            elif d_loss_fake >= 0.1:
                #### If the discriminator loss is already higher than the threshold, than Discriminator Trains normally
                d_loss_list.append(d_loss)
                d_loss.backward()
                optimizer_d.step()
                discri_train_counter = 0
            #### OPTIONAL #### : if you want to see the losses of the Discriminator, you can un-comment these
            print("d_loss_real =", d_loss_real)
            print("d_loss_fake =", d_loss_fake)
            print("d_loss =", d_loss)

            # Train Generator
            for i in range(3):
                #### In our case, our Generator is learning way too harder than the discriminator because of the
                #### non-uniform nature of the dataset we are using. Therefore we have training the generator "X" times
                #### more than the Discriminator Training.
                generator.zero_grad()
                # Adversarial loss (BCE)
                if i == 0:
                    ### at the fist iteration, we are using the fake data we already created on the
                    ### discriminator training part
                    trick_output = discriminator(fake_data)
                    print(trick_output)
                else:
                    #### after the first Training of the Generator, we have to create new fake data
                    #### to not train our generator with the same score again and again.
                    fake_data = generator(noisy_data)
                    trick_output = discriminator(fake_data)

                # Generator loss created with Binary Cross Entropy
                g_loss_adv = criterion_bce(trick_output, real_labels)

                #### OPTIONAL #### : In this case we have sometimes used the Mean Square Error to see if we
                #### are able to overfit the data.
                # Content loss (MSE): Ensure the generated spectrogram is close to the real spectrogram
                # g_loss_mse = criterion_mse(fake_data, real_data)

                #### OPTIONAL #### According to the aim, Mean Square Error result can be add to the loss of the Generator
                g_loss = g_loss_adv

                print("g_loss =", g_loss)
                # Content loss (MSE)
                # g_loss_mse = criterion_mse(fake_data, real_data)
                # print(g_loss_mse)

                if i == 2:
                    #### we have saved the loss result of the last training of generator training
                    g_loss_list.append(g_loss)
                g_loss.backward()
                optimizer_g.step()

        if epoch % 1 == 0:
            print(fake_data)
        # Logging for insight, not necessary for training
        if epoch % 1 == 0:  #### OPTIONAL #### : Log every "X" epochs
            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}"
            )
        if (epoch > 0) and (
            epoch % 1000 == 0
        ):  #### Each "X" Epoch, the models got saved to prevent loss of Trained Model
            try:
                torch.save(generator.state_dict(), Generator_Save_Path)
                torch.save(discriminator.state_dict(), Discriminator_Save_Path)
            except:
                print("couldn't save trained step models")
    # print(fake_data)


#### OPTIONAL #### :Loading the pre-trained models

# generator.load_state_dict(torch.load("/home/aksoyadnan/GAN_Files/Generator_MFCCLOG_New_trained_Finaltry4_segment05_6000_lr0008_NoBatchNormNoMSE_LeakyChange_2000Hz1File_DiscriminatorWBatchnorm_statedict"))
# discriminator.load_state_dict(torch.load("/home/aksoyadnan/GAN_Files/Discriminator_MFCCLOG_New_trained_Finaltry4_segment05_6000_lr0008_NoBatchNormNoMSE_LeakyChange_2000Hz1File_DiscriminatorWBatchnorm_statedict"))


#### Path of the save paths of both models
Generator_Backup_Save_Path = "../models/backups/generators/Backup_Generator_File_Name"
Discriminator_Backup_Save_Path = "../models/backups/discriminators/Backup_Discriminator_File_Name"

#### number of epochs
num_epochs = 1
# Call train function
train(
    generator,
    discriminator,
    dataloader,
    device,
    num_epochs=num_epochs,
    Generator_Save_Path=Generator_Backup_Save_Path,
    Discriminator_Save_Path=Discriminator_Backup_Save_Path,
)


Generator_Loss_Save_Path = "../results/Loss_Datas_To_Plot/Generator_Loss_Pickle_File"
Discriminator_Loss_Save_Path = "../results/Loss_Datas_To_Plot/Discriminator_Loss_Pickle_File"
with open(Generator_Loss_Save_Path, "wb") as fp:  # Pickling
    pickle.dump(g_loss_list, fp)
with open(Discriminator_Loss_Save_Path, "wb") as fp:  # Pickling
    pickle.dump(d_loss_list, fp)

#### The Save path of completed models
Completed_PreTrained_Generator_Path = "../models/generators/Backup_Generator_File_Name"
Completed_PreTrained_Discriminator_Path = "../models/discriminators/Backup_Discriminator_File_Name"

torch.save(generator.state_dict(), Completed_PreTrained_Generator_Path)
torch.save(discriminator.state_dict(), Completed_PreTrained_Discriminator_Path)


# torch.save(generator,"/home/aksoyadnan/GAN_Files/Generator_MFCCLOG_trained_segment05_6000_lr000008_LeakyChange_2000Hz1File_DiscriminatorWBatchnorm" )
# torch.save(discriminator,"/home/aksoyadnan/GAN_Files/Discriminator_MFCCLOG_trained_segment05_6000_lr000008_LeakyChange_2000Hz1File_DiscriminatorWBatchnorm")
