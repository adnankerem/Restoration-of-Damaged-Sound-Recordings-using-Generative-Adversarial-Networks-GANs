import torch
import Preprocess_Afterprocess_GAN_Combined as PAGAN

# Initialize dataset and dataloader
dataset = PAGAN.AudioDataset(
    clean_files=["../data/Raw_Sound_Files/Clean_Sound_File.wav"],
    noisy_files_list=["../data/Raw_Sound_Files/Clean_Sound_File.wav"],
    start_time=3,
    duration=2,
    segment_length=0.5,
    mfcc_bool=False,
    stft_bool=True,
    logscale_bool=True
)

# Create dataloader
dataloader = PAGAN.DataLoader(dataset, batch_size=1, shuffle=True)

# Save the dataloader
save_path = "../data/PreProcessed_Datasets/Dataset Filename" 
torch.save(dataloader, save_path)
print(f"Dataloader saved to {save_path}")
