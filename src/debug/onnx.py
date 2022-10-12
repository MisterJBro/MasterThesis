import timeit
from multiprocessing import freeze_support
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3*10*10, 10)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    freeze_support()

    # Init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Network().to(device)
    optimizer = optim.Adam(net.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()

    # Get MNIST trainset
    trainset = datasets.MNIST(
        root = 'data',
        train = True,
        transform = ToTensor(),
        download = True,
    )
    trainloader = DataLoader(trainset, batch_size=1000, shuffle=True, num_workers=1)

    # Training
    for epoch in range(0):
        for img, label in trainloader:
            optimizer.zero_grad()
            pred = net(img.to(device))
            loss = criterion(pred, label.to(device))
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch} Loss {loss.item():.03f}")

    # Save & load net
    #torch.save(net.state_dict(), 'net.pt')
    net.load_state_dict(torch.load('net.pt'))

    # Model to onnx
    net.cpu()
    net.eval()
    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(net, dummy_input, 'net.onnx', verbose=True, input_names = ['x'], output_names = ['y'],
        dynamic_axes={
            'x' : {0 : 'batch_size'},
            'y' : {0 : 'batch_size'}
        })

    # Create onnx inference session
    import onnxruntime as ort
    session = ort.InferenceSession('net.onnx', providers=['CPUExecutionProvider'])

    # Prediction
    with torch.no_grad():
        net.eval()
        input = trainset.data[0].reshape(1, 1, 28, 28).float()

        print(f"CPU Inference (PyTorch): {timeit.timeit(lambda: net(input), number=1000):.03f}s")
        print(f"CPU Inference (ONNX): {timeit.timeit(lambda: session.run(None, {'x': input.numpy()}), number=1000):.03f}s")

        py_out = net(input).cpu().numpy().argmax()
        ort_out = session.run(None, {'x': input.numpy()})[0].argmax()
        plt.imshow(trainset.data[0], cmap='gray')
        plt.title(f"Prediction (Pytorch): {py_out}  Prediction (ONNX): {ort_out} Truth: {trainset.targets[0]}")
        plt.show()