import argparse
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tqdm import tqdm
from autoencoder import MNISTAutoencoder
from config import MODEL_PATH, TEST_IMG
from vae import MNISTGenerative
import cv2

DATASET_ROOT = "C:/Users/ikerv/Documents/VS_Code_repos/MNIST_generative/data/"
DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

BATCH_SIZE = 512
LR = 3e-4
NUM_EPOCHS = 10

MU = 0.5
SIGMA = 1.0

def preview_dataset(train_data) -> None:
    figure = plt.figure(figsize=(8,8))
    cols, rows = 3, 3
    for i in range(1, cols*rows+1):
        idx = torch.randint(len(train_data), size=(1,),).item()
        img, label = train_data[idx]
        figure.add_subplot(cols, rows, i)
        plt.title(label)
        plt.axis("off")
        print(img)
        plt.imshow(img, cmap="gray")
    plt.show()

def compare_images(img0, img1):
    figure = plt.figure(figsize=(8,8))
    cols, rows = 2, 1
    figure.add_subplot(cols, rows, 1)
    plt.title("Img0")
    plt.axis("off")
    plt.imshow(img0, cmap="gray")
    figure.add_subplot(cols, rows, 2)
    plt.title("Img1")
    plt.axis("off")
    plt.imshow(img1, cmap="gray")
    plt.show()

def show_img(img):
    plt.figure(figsize=(8,8))
    plt.imshow(img, cmap="gray")
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-train', action='store_true')
    args = parser.parse_args()

    if args.train is True:
        train_data = datasets.MNIST(DATASET_ROOT, train=True, download= True, transform=ToTensor())
        test_data = datasets.MNIST(DATASET_ROOT, train=False, download=True, transform=ToTensor())

        train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

        model = MNISTAutoencoder().to(DEVICE)
        model.train(train_dataloader, device=DEVICE)
        

    else:
        model = torch.load(MODEL_PATH).to(DEVICE)

        img = cv2.imread(TEST_IMG, cv2.IMREAD_GRAYSCALE)
        img = torch.Tensor(img / np.max(img))
        image0 = torch.clone(img)
        image0 = torch.squeeze(image0)
        
        img = img.to(DEVICE)

        x_gen = model.forward(torch.unsqueeze(img, 0))
        
        image1 = x_gen.to("cpu")
        image1 = image1.detach()
        image1 = image1.squeeze()
        compare_images(image0, image1)


    