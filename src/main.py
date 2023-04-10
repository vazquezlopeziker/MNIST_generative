import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from autoencoder import MNISTAutoencoder
from config import BATCH_SIZE, DATASET_ROOT, DEVICE, MODEL_PATH, TEST_IMG

from utils import compare_images


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
        model.train(train_dataloader, test_dataloader)
        
    else:
        model = torch.load(MODEL_PATH).to(DEVICE)

        train_data = datasets.MNIST(DATASET_ROOT, train=True, download= True, transform=ToTensor())
        img, _ = train_data[0]

        image0 = torch.clone(img)
        image0 = torch.squeeze(image0)
        
        img = img.to(DEVICE)

        x_gen = model.forward(torch.unsqueeze(img, 0))
        
        image1 = x_gen.to("cpu")
        image1 = image1.detach()
        image1 = image1.squeeze()

        compare_images(image0, image1)

        


    