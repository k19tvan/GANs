import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, z_dim=100, img_channels=3, img_size=128):
        super(Generator, self).__init__()

        self.img_size = img_size
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, img_channels * img_size * img_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 3, self.img_size, self.img_size)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, img_size=128):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_channels * img_size * img_size, 1024),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)