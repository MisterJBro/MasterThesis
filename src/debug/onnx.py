import timeit
from multiprocessing import freeze_support
from torchinfo import summary
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

        self.body = nn.Sequential(
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
        )
        self.head1 = nn.Linear(3*10*10, 10)
        self.head2 = nn.Linear(3*10*10, 1)

    def forward(self, x):
        x = self.body(x)
        p = self.head1(x)
        v = self.head2(x)
        return p, v

if __name__ == '__main__':
    freeze_support()

    # Init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Network().to(device)
    optimizer = optim.Adam(net.parameters(), lr=2e-4)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.HuberLoss()

    # Summary
    summary(net, input_size=(1000, 1, 28, 28))

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
            label = label.to(device)
            pred, pred2 = net(img.to(device))
            loss1 = criterion1(pred, label)
            loss2 = criterion2(pred2, label.float())
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch} Loss {loss.item():.03f} Loss1 {loss1.item():.02f} Loss2 {loss2.item():.02f}")

    # Save & load net
    #torch.save(net.state_dict(), 'net.pt')
    net.load_state_dict(torch.load('net.pt'))

    # Model to TorchScript
    net.cpu()
    net.eval()
    dummy_input = torch.randn(1, 1, 28, 28)
    net_jit = torch.jit.script(net, example_inputs=[[dummy_input]])

    # Model to onnx
    torch.onnx.export(net_jit, dummy_input, 'net.onnx', input_names = ['x'], output_names = ['y'],
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
        print(f"CPU Inference (TorchScript): {timeit.timeit(lambda: net_jit(input), number=1000):.03f}s")
        print(f"CPU Inference (ONNX): {timeit.timeit(lambda: session.run(None, {'x': input.numpy()}), number=1000):.03f}s")

        print(net(input))
        print(net_jit(input))
        print(session.run(None, {'x': input.numpy()}))
        py_out = net(input)[0].cpu().numpy().argmax()
        ort_out = session.run(None, {'x': input.numpy()})[0].argmax()
        plt.imshow(trainset.data[0], cmap='gray')
        plt.title(f"Prediction (Pytorch): {py_out}  Prediction (ONNX): {ort_out} Truth: {trainset.targets[0]}")
        plt.show()