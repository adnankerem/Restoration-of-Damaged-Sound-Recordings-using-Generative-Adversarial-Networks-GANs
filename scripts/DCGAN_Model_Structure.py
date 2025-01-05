#!/home/aksoyadnan/miniconda3/envs/myenv/bin/python3.12
import torch
import torch.nn as nn

def calculate_output_size(input_size, kernel_size, stride, padding):
    """
    Calculate the output size of a layer given its parameters.
    
    Args:
        input_size (int): Input size (e.g., frequency bins or frame rate).
        kernel_size (int): Kernel size of the layer.
        stride (int): Stride of the layer.
        padding (int): Padding applied to the input.
    
    Returns:
        int: Output size after the layer.
    """
    return (input_size + 2 * padding - kernel_size) // stride + 1


class Generator(nn.Module):
    """
    Generator network for DCGAN.
    """
    def __init__(self, frame_rate: int, channel_count: int, freq_bins: int):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: [batch_size, channel_count, freq_bins, frame_rate]
            nn.ConvTranspose2d(channel_count, 128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), bias=False),
            nn.LayerNorm([128, freq_bins, frame_rate]),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.001),
            
            nn.ConvTranspose2d(128, 256, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), bias=False),
            nn.LayerNorm([256, freq_bins, frame_rate]),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.001),
            
            nn.ConvTranspose2d(256, 256, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), bias=False),
            nn.LayerNorm([256, freq_bins, frame_rate]),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.001),
            
            nn.ConvTranspose2d(256, channel_count, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), bias=False),
            nn.Tanh()  # Normalize output to match the spectrogram value range
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    """
    Discriminator network for DCGAN.
    """
    def __init__(self, frame_rate: int, channel_count: int, freq_bins: int):
        super(Discriminator, self).__init__()
        
        # Calculate output dimensions for each convolutional layer
        freq_1 = calculate_output_size(freq_bins, kernel_size=3, stride=2, padding=1)
        frame_1 = calculate_output_size(frame_rate, kernel_size=5, stride=2, padding=2)
        
        freq_2 = calculate_output_size(freq_1, kernel_size=3, stride=2, padding=1)
        frame_2 = calculate_output_size(frame_1, kernel_size=5, stride=2, padding=2)
        
        freq_3 = calculate_output_size(freq_2, kernel_size=3, stride=2, padding=1)
        frame_3 = calculate_output_size(frame_2, kernel_size=5, stride=2, padding=2)
        
        self.main = nn.Sequential(
            nn.Conv2d(channel_count, 64, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2), bias=False),
            nn.LayerNorm([64, freq_1, frame_1]),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.7),
            
            nn.Conv2d(64, 128, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2), bias=False),
            nn.LayerNorm([128, freq_2, frame_2]),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.7),
            
            nn.Conv2d(128, 128, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2), bias=False),
            nn.LayerNorm([128, freq_3, frame_3]),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.7),
            
            nn.Flatten(),
            nn.Linear(128 * freq_3 * frame_3, 1)  # Fully connected layer for binary classification
        )

    def forward(self, x):
        return self.main(x)


def init_weights(m):
    """
    Initialize weights for Conv and Linear layers in the network.
    Uses Xavier initialization for stability during training.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.2)
        nn.init.constant_(m.bias.data, 0)
