
import torch
import numpy as np

from train_model import TinyCNN


model = TinyCNN()

model.load_state_dict(
    torch.load(
        "tiny_cnn.pth",
        map_location="cpu"
    )
)

model.eval()


# ------------------------------------------------------------
# CONVOLUTION WEIGHTS
# Shape:
# [8 filters, 1 channel, 3, 3]
# ------------------------------------------------------------

conv_weights = (
    model.conv.weight
    .detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)


# ------------------------------------------------------------
# FULLY CONNECTED WEIGHTS
# PyTorch stores:
# [10 outputs, 1352 inputs]
# ------------------------------------------------------------

fc_weights = (
    model.fc.weight
    .detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)


conv_weights.tofile(
    "conv_weights.bin"
)

fc_weights.tofile(
    "fc_weights.bin"
)


print(
    "Conv weights:",
    conv_weights.shape
)

print(
    "FC weights:",
    fc_weights.shape
)

print(
    "Export complete."
)
