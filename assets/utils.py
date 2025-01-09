import torch 
import matplotlib.pyplot as plt
import copy
import PIL 
    
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

