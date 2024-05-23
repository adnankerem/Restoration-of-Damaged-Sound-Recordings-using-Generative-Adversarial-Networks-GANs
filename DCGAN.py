import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: [2, 1025, 188]
            nn.ConvTranspose2d(2, 128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2),bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 256, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2),bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ConvTranspose2d(256, 256, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2),bias=False),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            nn.ConvTranspose2d(256, 2, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2),bias=False),
            nn.Tanh()  # Normalize output to match the range of spectrogram values
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: [1, 1025, 188]
            nn.Conv2d(2, 64, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2),bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2),bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2),bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(396288, 1),  # Adjust the size according to the output of the last conv layer
            
        )

    def forward(self, x):
        return self.main(x)
# def init_weights(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1 or classname.find('Linear') != -1:
#         if hasattr(m, 'weight') and m.weight is not None:
#             nn.init.kaiming_uniform_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
#         if hasattr(m, 'bias') and m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)

# custom weights initialization called on ``netG`` and ``netD``

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.xavier_uniform_(m.weight.data)
    if classname.find('Linear') != -1:
        #nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.2)
        nn.init.constant_(m.bias.data, 0)