import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import time
from src.env.hex import HexEnv
from src.networks.residual import HexPolicy
from src.train.config import create_config
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load policy
size = 6
env = HexEnv(size)
config = create_config({
    "env": env,
})
policy = HexPolicy(config)
policy.load("checkpoints/policy_hex_6x6.pt")
policy.eval()
device = torch.device("cuda:0")

"""class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(2, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.body(x)
net = Network()
net.to(device)
net.eval()"""

x = torch.rand(128, 2, size, size, device=device)
policy(x)
time.sleep(10.0)

# Test batch size
split_exp = 16
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 196, 256, 320, 384, 448, 502, 512, 520, 528, 576, 640, 704, 768, 1024, 1536, 2048, 3072, 4096, 5120]#[x for x in range(16, 768, 1)]
throughputs = [[] for _ in range(len(batch_sizes))]
for _ in range(10):
    for i, batch_size in enumerate(batch_sizes):
        try:
            # Get throughput
            with torch.no_grad():
                x = torch.rand(batch_size, 2, size, size, device=device)
                torch.cuda.synchronize(device)
                start = time.time()
                policy(x)
                torch.cuda.synchronize(device)
                end = time.time()
                throughput = batch_size / (end - start)
                print(f"Batch size: {batch_size} \t {throughput:.02f} Sample/s")
            throughputs[i].append(throughput)
        except Exception as e:
            print(f"Batch:{batch_size} {e}")
            break
    print(np.round(throughputs))

# Get median
throughputs = [np.median(x) for x in throughputs]

# Get highest throughput batch size
max_throughput = np.max(throughputs)
max_batch_size = batch_sizes[np.argmax(throughputs)]
print(f"Max throughput: {max_throughput:.02f} Sample/s at batch size: {max_batch_size}")

# Plot
sns.set_theme()
plt.plot(np.array(batch_sizes)[:len(throughputs)], throughputs)
plt.xlabel("Batch Size")
plt.ylabel("Throughput (Samples/s)")
plt.show()
