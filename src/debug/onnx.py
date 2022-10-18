from copy import deepcopy
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
import onnx
import onnxruntime as ort
import onnxoptimizer

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class ResBlock(nn.Module):
    """ Residual Block with Skip Connection, just like ResNet. """
    def __init__(self, num_filters, kernel_size):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
        )

    def forward(self, x):
        return F.relu(x + self.layers(x))

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            *[ResBlock(32, 3) for _ in range(5)],
            nn.Flatten(1, -1),
        )
        self.head1 = nn.Linear(4608, 10)
        self.head2 = nn.Linear(4608, 1)

    def forward(self, x, mode):
        x = self.body(x)
        if mode == 1:
            p = self.head1(x)
            v = torch.zeros((x.shape[0],), device=x.device)
        elif mode == 2:
            p = torch.zeros((x.shape[0], 10), device=x.device)
            v = self.head2(x).reshape(-1)
        else:
            p = self.head1(x)
            v = self.head2(x).reshape(-1)
        return p, v

if __name__ == '__main__':
    freeze_support()
    # Pytorch Optimizations
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high") # https://pytorch.org/docs/master/generated/torch.set_float32_matmul_precision.html?highlight=precision#torch.set_float32_matmul_precision

    # Init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Network().to(device)
    optimizer = optim.Adam(net.parameters(), lr=2e-4)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.HuberLoss()
    dummy_input = [torch.randn(1, 1, 28, 28, dtype=torch.float32), torch.tensor(0, dtype=torch.int32)]

    # Summary
    summary(net, input_data=dummy_input)

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
            optimizer.zero_grad(set_to_none=True)
            label = label.to(device)
            pred, pred2 = net(img.to(device), torch.tensor(0))
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
    net_jit = torch.jit.script(net, example_inputs=[dummy_input])

    # Model to onnx
    torch.onnx.export(net_jit, dummy_input, 'net.onnx', opset_version=15, input_names=['x', 'mode'], output_names=['p', 'v'],
        dynamic_axes={
            'x' : {0 : 'batch_size'},
            'p' : {0 : 'batch_size'},
            'v' : {0 : 'batch_size'},
        })

    # Optimize model (so we do not get any warning later)
    onnx_model = onnx.load('net.onnx')
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    optimized_model = onnxoptimizer.optimize(onnx_model, passes)

    onnx.save(optimized_model, 'net.onnx')

    # Create onnx inference session
    session = ort.InferenceSession('net.onnx', providers=['CPUExecutionProvider'])
    session_cuda = ort.InferenceSession('net.onnx', providers=['CUDAExecutionProvider'])
    session_tensorrt = ort.InferenceSession('net.onnx', providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])

    # Prediction
    with torch.no_grad():
        net.eval()
        batch_size = 32
        input = trainset.data[0].reshape(1, 1, 28, 28).float()
        input_batch = trainset.data[:batch_size].reshape(batch_size, 1, 28, 28).float()
        mode = torch.tensor(0, dtype=torch.int32)

        # CPU
        print(f"CPU Inference (PyTorch): {timeit.timeit(lambda: net(input_batch, mode), number=1000):.03f}s")
        print(f"CPU Inference (TorchScript): {timeit.timeit(lambda: net_jit(input_batch, mode), number=1000):.03f}s")
        print(f"CPU Inference (ONNX): {timeit.timeit(lambda: session.run(None, {'x': input_batch.numpy(), 'mode': mode.numpy()}), number=1000):.03f}s")

        # GPU
        dummy_input_gpu = [dummy_input[0].cuda(), dummy_input[1]]
        net_gpu = deepcopy(net).cuda()
        net_jit_gpu = torch.jit.script(net_gpu, example_inputs=[dummy_input_gpu]).cuda()
        input_gpu = input_batch.cuda()
        torch.cuda.synchronize()

        print(f"GPU Inference (PyTorch): {timeit.timeit(lambda: net_gpu(input_batch.cuda(), mode)[0].cpu() + 1, number=1000):.03f}s")
        with torch.cuda.amp.autocast():
            print(f"GPU Inference (PyTorch AMP): {timeit.timeit(lambda: net_gpu(input_batch.cuda(), mode)[0].cpu() + 1, number=1000):.03f}s")
        print(f"GPU Inference (TorchScript): {timeit.timeit(lambda: net_jit_gpu(input_batch.cuda(), mode)[0].cpu() + 1, number=1000):.03f}s")
        print(f"GPU Inference (ONNX CUDA): {timeit.timeit(lambda: session_cuda.run(None, {'x': input_batch.numpy(), 'mode': mode.numpy()})[0] + 1, number=1000):.03f}s")
        print(f"GPU Inference (ONNX TensorRT): {timeit.timeit(lambda: session_tensorrt.run(None, {'x': input_batch.numpy(), 'mode': mode.numpy()})[0] + 1, number=1000):.03f}s")

        print(net(input, mode))
        with torch.cuda.amp.autocast():
            print(net(input, mode))
        print(session.run(None, {'x': input.numpy(), 'mode': mode.numpy()}))
        py_out = net(input, mode)[0].cpu().numpy().argmax()
        ort_out = session.run(None, {'x': input.numpy(), 'mode': mode.numpy()})[0].argmax()
        plt.imshow(trainset.data[0], cmap='gray')
        plt.title(f"Prediction (Pytorch): {py_out}  Prediction (ONNX): {ort_out} Truth: {trainset.targets[0]}")
        plt.show()