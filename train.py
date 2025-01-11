import torch 
import torch.nn as nn
import torch.optim.adam
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pathlib import Path
from assets.customdatasets import CustomDataset
from models.model_cnn import Discriminator, Generator
import matplotlib.pyplot as plt
import torchvision
import argparse
from tqdm import tqdm
import cv2
from pathlib import Path
import copy 

""" PipeLine: Hyperparameters -> Transforms -> Dataloader -> Model -> Training """

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mean = torch.tensor([0.485, 0.456, 0.406])
# std = torch.tensor([0.229, 0.224, 0.225])

mean = torch.tensor([0.5, 0.5, 0.5])    
std = torch.tensor([0.5, 0.5, 0.5])

def imshow(image):

    device = image.device

    cur_mean = mean.view(-1, 1, 1)
    cur_std = std.view(-1, 1, 1)

    img = copy.deepcopy(image)
    img = img.mul(cur_std.to(device)).add(cur_mean.to(device))
    img = img.clamp(0, 1)

    img = img.permute(1, 2, 0).cpu().numpy()

    plt.imshow(img)
    plt.axis('off')
    plt.show()


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
    parser.add_argument('--latent_dim', type=int, default=70)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--save_period', type=int, default=1)
    
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
save_period = args.save_period

""" Transforms """

transform = transforms.Compose([
    transforms.Resize(img_size),
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
discriminator = Discriminator().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

adversarial_loss = nn.BCELoss()

""" Training """

def count_files(dir):
    return len(list(dir.glob("*")))

def training():

    run_dir = Path("run")
    run_dir.mkdir(parents=True, exist_ok=True)

    cnt = count_files(run_dir)

    for epoch in range(num_epochs):
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for i, imgs in enumerate(dataloader):

                real_imgs = imgs.to(device)
                batch_size = real_imgs.size(0)

                valid = torch.ones(batch_size, 1).to(device)
                fake = torch.zeros(batch_size, 1).to(device)

                """ Generator Training """
                optimizer_D.zero_grad()
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_imgs = generator(z).to(device).detach()

                # print(valid.shape)

                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(fake_imgs), fake)
                
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()

                optimizer_D.step()

                """ Discriminator Traning"""
                optimizer_G.zero_grad()

                z = torch.randn(batch_size, latent_dim).to(device)
                fake_imgs = generator(z).to(device)

                g_loss = adversarial_loss(discriminator(fake_imgs), valid)

                g_loss.backward()

                optimizer_G.step()

                pbar.update(1)

                if (i + 1) % 100 == 0:
                    pbar.set_postfix(
                        d_loss=d_loss.item(),
                        g_loss=g_loss.item()
                    )

            if (epoch + 1) % save_period == 0:
                with torch.no_grad():
                    z = torch.randn(16, latent_dim).to(device)

                    generated_imgs = generator(z).detach().cpu()
                    generated_imgs = (generated_imgs + 1) / 2

                    grid = torchvision.utils.make_grid(generated_imgs, nrow=4, normalize=True)

                    img = grid.permute(1, 2, 0).numpy() * 255

                    cur_dir = run_dir/Path(f"run{cnt + 1}")
                    cur_dir.mkdir(parents=True, exist_ok=True)

                    image_dir = cur_dir/Path(f"Epoch{epoch + 1}.png")

                    print(f"Save image at {image_dir}")
                    cv2.imwrite(image_dir, img)

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
    print(f"save_period = {save_period}")

    print()

    print("Training", "-"*20)
    training()

