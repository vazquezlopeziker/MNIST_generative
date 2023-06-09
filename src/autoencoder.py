import torch
from torch import nn
from tqdm import tqdm

from config import LR, MODEL_PATH, NUM_EPOCHS, DEVICE


class MNISTAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(28*28, 300)
        self.linear2 = nn.Linear(300, 81)

        self.linear3 = nn.Linear(81, 300)
        self.linear4 = nn.Linear(300, 28*28)
    
    def encode(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = nn.ReLU()(x)

        x = self.linear2(x)
        x = nn.ReLU()(x)

        return x
    
    def decode(self, x):
        x = self.linear3(x)
        x = nn.ReLU()(x)

        x = self.linear4(x)
        x = x.view(-1, 1, 28,28)
        x = torch.sigmoid(x)

        return x

    def forward(self, x):
        x = self.encode(x)  
        x = self.decode(x)

        return x
    
    def train(self, train_dataloader, test_dataloader):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        loss_fn = torch.nn.BCELoss()

        for _ in range(NUM_EPOCHS):
            loop = tqdm(enumerate(train_dataloader))
            for _, (x, _) in loop:
                x = x.to(DEVICE)
                x_gen = self(x)

                loss = loss_fn(x_gen, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loop.set_postfix(loss=loss.item())
            
            test_loss = torch.tensor([0.0], requires_grad=True)
            test_loss = test_loss.to(DEVICE)
            for i, (t,_) in enumerate(test_dataloader):
                t = t.to(DEVICE)
                t_gen = self(t)
                test_loss += loss_fn(t_gen, t)
            test_loss = test_loss/ i
            print(f"test_loss={test_loss.item()}")

        torch.save(self,MODEL_PATH)