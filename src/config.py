import torch


NUM_EPOCHS = 50
BATCH_SIZE = 512
LR = 3e-4

TEST_IMG = "data/test_paint.png"
MODEL_PATH = "results/autoencoder/model"
DATASET_ROOT = "C:/Users/ikerv/Documents/VS_Code_repos/MNIST_generative/data/"

DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )