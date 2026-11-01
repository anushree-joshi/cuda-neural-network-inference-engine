
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

class TinyCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            bias=False
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.fc = nn.Linear(
            8 * 13 * 13,
            10,
            bias=False
        )


    def forward(self, x):

        x = self.conv(x)

        x = self.relu(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Training on:", device)


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

model = TinyCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


for epoch in range(2):

    model.train()

    total_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    print(
        f"Epoch {epoch + 1}, "
        f"Loss: {total_loss / len(train_loader):.4f}"
    )


# ------------------------------------------------------------
# TEST ACCURACY
# ------------------------------------------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


accuracy = 100 * correct / total

print(
    f"Test accuracy: {accuracy:.2f}%"
)


# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------

# ------------------------------------------------------------
# SAVE MODEL + EXPORT WEIGHTS
# ------------------------------------------------------------

import numpy as np

# Save normal PyTorch model
torch.save(
    model.state_dict(),
    "tiny_cnn.pth"
)

print("Saved tiny_cnn.pth")


# Export convolution weights for our C++/CUDA engine
model.conv.weight.detach().cpu().numpy().astype(
    np.float32
).tofile("conv_weights.bin")


# Export fully-connected weights for our C++/CUDA engine
model.fc.weight.detach().cpu().numpy().astype(
    np.float32
).tofile("fc_weights.bin")


print("Exported conv_weights.bin")
print("Exported fc_weights.bin")
