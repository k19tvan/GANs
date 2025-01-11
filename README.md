# GANs - Generative Adversarial Network

Implementation of a Generative Adversarial Network (GAN).

<div align="center">
    <img src="epoch24.png" alt="Image Generated At Epoch 24" title="Image Generated At Epoch 24">
</div>

---

### Table of Contents

- [Installation](#installation)
  - [Environment Setup](#environment-setup)
  - [Dataset Setup](#dataset-setup)
- [Training](#training)
  - [Default Hyperparameters](#default-hyperparameters)
  - [Custom Hyperparameters](#custom-hyperparameters)

---

### Installation

#### Environment Setup
To set up the environment, follow these steps:
```bash
conda create -n gans python=3.11
conda activate gans
git clone https://github.com/k19tvan/DCC_Basic
cd GANs
pip install -r requirements.txt
```

#### Dataset Setup
Ensure the dataset is properly configured:
```bash
python data.py
sudo apt install unzip
unzip -o train.zip Cats_faces_64_x_64.zip
```

---

### Training

#### Default Hyperparameters
To train the model with default hyperparameters, run the following command:
```bash
python train.py
```

| **Hyperparameter** | **Type** | **Default Value** | **Description**                        |
|---------------------|----------|-------------------|----------------------------------------|
| `batch_size`        | `int`    | 100               | Number of samples in each batch.       |
| `shuffle`           | `str`    | `'True'`          | Whether to shuffle the data after each epoch. |
| `num_workers`       | `int`    | 0                 | Number of worker threads for the DataLoader. |
| `num_epochs`        | `int`    | 25                | Number of training epochs.             |
| `lr`                | `float`  | 0.0002            | Learning rate for the optimizer.       |
| `beta1`             | `float`  | 0.5               | Beta1 parameter for Adam optimizer.    |
| `beta2`             | `float`  | 0.999             | Beta2 parameter for Adam optimizer.    |
| `latent_dim`        | `int`    | 100               | Dimensionality of the latent space.    |
| `img_size`          | `int`    | 64                | Input image size.                      |
| `save_period`       | `int`    | 1                 | Period to save the model after each epoch. |

#### Custom Hyperparameters
To train the model with custom hyperparameters, use the following syntax:
```bash
python train.py --batch_size 100 --num_epochs 20 --img_size 128 ...
```
Replace the `...` with additional arguments as needed to customize your training process.

