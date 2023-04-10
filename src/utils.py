from matplotlib import pyplot as plt
import torch


def preview_dataset(train_data) -> None:
    figure = plt.figure(figsize=(8,8))
    cols, rows = 3, 3
    for i in range(1, cols*rows+1):
        idx = torch.randint(len(train_data), size=(1,),).item()
        img, label = train_data[idx]
        figure.add_subplot(cols, rows, i)
        plt.title(label)
        plt.axis("off")
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

