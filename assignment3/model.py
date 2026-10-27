"""CAP 5410 - Assignment 3: CNN model (LeNet-5 style) for MNIST."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """Two convolutional layers, two max-pooling layers and three fully connected layers.

    Shapes for one 1 x 28 x 28 MNIST image:
        conv1 (5x5, padding 2) -> 6 x 28 x 28 -> max-pool 2x2 -> 6 x 14 x 14
        conv2 (5x5)            -> 16 x 10 x 10 -> max-pool 2x2 -> 16 x 5 x 5
        flatten                -> 400
        fc1 -> 120, fc2 -> 84, fc3 -> 10 class scores (logits)
    """

    def __init__(self):
        super(Net, self).__init__()
        # LAYERS
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Feed-forward
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)          # no softmax: CrossEntropyLoss applies log-softmax itself
        return x


def layer_table(model, input_shape=(1, 1, 28, 28)):
    """Return one row per layer: name, output shape and trainable parameters.

    Output shapes are found by passing a dummy image through the network
    with forward hooks; parameter counts come from the layer's tensors.
    """
    rows = []
    hooks = []

    def hook(name):
        def fn(module, inp, out):
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            rows.append((name, tuple(inp[0].shape[1:]), tuple(out.shape[1:]), n_params))
        return fn

    for name, module in model.named_children():
        hooks.append(module.register_forward_hook(hook(name)))
    with torch.no_grad():
        model(torch.zeros(input_shape))
    for h in hooks:
        h.remove()
    return rows


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
