import torch.nn as nn
import torch
from math import *
        
class Generator(nn.Module):

    """ 
    Tensor Size In After Generator's Layers (no batch_size)

    - initial: (1, latent_dim)
    - dense: (img_size // 16) * (img_size // 16) * 1024)
    - view: (1024, img_size // 16, img_size // 16)
    - deconv1: (512, img_size // 8, img_size // 8)
    - deconv2: (256, img_size // 4, img_size // 4)
    - deconv3: (128, img_size // 2, img_size // 2)
    - deconv4: (3, img_size, img_size) 

    * Output (with batch_size): tensor(1, 3, img_size, img_size)

    """

    def __init__(self, latent_dim, img_size=64):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_size = img_size

        self.dense = nn.Linear(latent_dim, (img_size // 16) * (img_size // 16) * 1024) 
        """ View """

        self.deconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=512)

        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=256)

        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)

        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=64)

        self.relu = nn.ReLU()

    def forward(self, z):
        z  = self.dense(z)
        z = z.view(z.size(0), 1024, self.img_size // 16, self.img_size // 16)

        z = self.relu(self.bn1(self.deconv1(z)))
        z = self.relu(self.bn2(self.deconv2(z)))
        z = self.relu(self.bn3(self.deconv3(z)))
        z = self.deconv4(z)

        z = torch.tanh(z)

        return z

class Discriminator(nn.Module):

    """ 
    Tensor Size In After Discriminator's Layers (no batch_size)

    - initial: (3, img_size, img_size)
    - conv1: (128, img_size // 2, img_size // 2)
    - conv2: (256, img_size // 4, img_size // 4)
    - conv3: (512, img_size // 8, img_size // 8)
    - conv4: (1024, img_size // 16, img_size / 1)
    - gap: (1024)
    - tanh: (1024)
    
    * Output (with bath_size): tensor(1, 1024)
    
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(512)

        self.conv4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(1024)

        self.conv5 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=4, stride=2, padding=1)

        self.dropout = nn.Dropout(0.3)
        self.lkrelu = nn.LeakyReLU(0.2)
        self.linear = nn.Linear(2048, 1)

        
    def forward(self, img):
        
        img = self.dropout(self.lkrelu(self.bn1(self.conv1(img))))
        img = self.dropout(self.lkrelu(self.bn2(self.conv2(img))))
        img = self.dropout(self.lkrelu(self.bn3(self.conv3(img))))
        img = self.bn4(self.conv4(img))

        """ Minibatch Standard Deviation """

        std = torch.std(img, dim=0, keepdim=True)
        std_mean = std.mean()
        std_map = std_mean.expand(img.shape)
        img = torch.cat([img, std_map], dim=1)

        img = torch.mean(img, dim=(2, 3))
        img = self.linear(img)
        img = torch.sigmoid(img)

        return img

