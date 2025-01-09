import torch 
import torch.nn as nn
import torch.optim.adam
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pathlib import Path
from assets.utils import mean, std, imshow
from assets.customdatasets import CustomDataset
from assets.model import Discriminator, Generator
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import argparse
from tqdm import tqdm

""" PipeLine: Hyperparameters -> Transforms -> Dataloader -> Model -> Training """

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyper Parameter """

def parse_args():
    parser = argparse.ArgumentParser(description='DogCatClassificationTraining')

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--shuffle', type=str, default='True')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default='0.5')
    parser.add_argument('--beta2', type=float, default='0.999')
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--img_size', type=int, default=128)
    
    return parser.parse_args()

args = parse_args()
batch_size = args.batch_size
shuffle = args.shuffle == 'True'
num_workers = args.num_workers
num_epochs = args.num_epochs
lr = args.lr
beta1 = args.beta1
beta2 = args.beta2
latent_dim = args.latent_dim
img_size = args.img_size

""" Transforms """

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

""" Dataloader """

data_dir = Path('cats')  
dataset = CustomDataset(root_dir=data_dir, transform=transform)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

imageIter = iter(dataloader)
images = imageIter.__next__()

for i, image in enumerate(images):
    imshow(image)
    if i >= 2: break

""" Model """

generator = Generator(latent_dim, img_size=img_size).to(device)
discriminator = Discriminator(img_size=img_size).to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

adversarial_loss = nn.BCELoss()

""" Training """

def training():
    for epoch in range(num_epochs):
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for i, imgs in enumerate(dataloader):

                real_imgs = imgs.to(device)
                batch_size = real_imgs.size(0)
                valid = torch.ones(batch_size, 1).to(device)
                fake = torch.zeros(batch_size, 1).to(device)

                optimizer_D.zero_grad()
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(generator(torch.randn(batch_size, 100).to(device)).detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()

                optimizer_G.zero_grad()
                g_loss = adversarial_loss(discriminator(generator(torch.randn(batch_size, 100).to(device))), valid)
                g_loss.backward()
                optimizer_G.step()

                pbar.update(1)

                if (i + 1) % 100 == 0:
                    pbar.set_postfix(
                        d_loss=d_loss.item(),
                        g_loss=g_loss.item()
                    )

            if epoch + 1 == num_epochs: 
                with torch.no_grad():
                    z = torch.randn(16, 100).to(device)
                    generated_imgs = generator(z).detach().cpu()
                    grid = torchvision.utils.make_grid(generated_imgs, nrow=4, normalize=True)
                    plt.imshow(np.transpose(grid, (1, 2, 0)))
                    plt.title(f"Epoch {epoch}")
                    plt.axis('off')
                    plt.show()

if __name__ == '__main__':

    print("\nHyperparameters", "-"*20)
    print(f"batch_size = {batch_size}")
    print(f"shuffle = {shuffle}")
    print(f"num_workers = {num_workers}")
    print(f"num_epochs = {num_epochs}")
    print(f"lr = {lr}")
    print(f"beta1 = {beta1}")
    print(f"beta2 = {beta2}")
    print(f"latent_dim = {latent_dim}")
    print(f"img_size = {img_size}")

    print()

    print("Training", "-"*20)
    training()

